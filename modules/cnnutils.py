#!/usr/bin/env python

import os,sys
import numpy
import base64
import theano
from theano.tensor.signal import downsample
import theano.tensor as T
import copy
from modules.utils.fsutils import DataFile

def get_model_params(model):
    params = []
    for param in model.params:
        params.append(param.get_value())
    return params

def set_model_params(model, params):
    for (m,p) in zip(model.params, params):
        m.set_value(p)

def get_cost_function(cf, cnn, y):
    if cf == 'negative_log_likelihood':
        return cnn.negative_log_likelihood(y)
    elif cf == 'cross_entropy':
        return cnn.cross_entropy(y)
    elif cf == 'least_squares_error':
        return cnn.least_squares_error(y)
    else:
        return None

def relu(x):
    return (x * (x > 0))

def get_activation(act_name):
    if act_name == 'relu':
        return relu
    elif act_name == 'sigmoid':
        return T.nnet.sigmoid
    elif act_name == 'tanh':
        return T.tanh
    elif act_name == 'softmax':
        return T.nnet.softmax
    else:
        return None

class CNNDataFileIter(object):
    def __init__(self, logger, fpath, reshape_dims, field_nums=[0,1], sep='|', dtype=numpy.float32, huge_data=False, size=0, max_buffer_data=1):
        self.logger = logger
        self.cnt = 0
        self.buf_ind = 0
        self.buf_cnt = 0
        self.size = size
        self.fp = DataFile(fpath, 'r', self.logger).get_fp()
        self.field_nums = field_nums
        self.sep = sep
        self.reshape_dims = reshape_dims
        self.dtype = dtype
        self.huge_data = huge_data
        self.datax = None
        self.datay = None
        self.max_buffer_data = max_buffer_data
        if self.huge_data and self.max_buffer_data <= 0:
            raise Exception('invalid value for max buffer data %d' % self.max_buffer_data)
        
        #count number of lines in file if reqd
        if self.size <= 0:
            self.size = 0
            for line in self.fp:
                self.size += 1
            self.fp.seek(0, 0)

        if not self.huge_data or self.max_buffer_data > 0:
            self.num_read = self.size
            if self.huge_data and self.max_buffer_data > 0:
                self.num_read = self.max_buffer_data
            self.datax = numpy.zeros([self.num_read]+self.reshape_dims[0], dtype=self.dtype)
            self.datay = numpy.zeros([self.num_read]+([] if (self.reshape_dims[1]==1) else self.reshape_dims[1]), dtype=self.dtype)
            self.reload_buffer(start=0)
            self.cnt = 0
            if not self.huge_data:
                self.permut = numpy.random.permutation(self.datay.shape[0])
                self.fp.close()
        self.logger.info('initialised iterator for file %s, found %d lines' % (fpath, self.size))

    def get_size(self):
        return self.size

    def shuffle_and_reset(self):
        if self.huge_data:
            self.reload_buffer(start=0)
        else:
            self.permut = numpy.random.permutation(self.datay.shape[0])
        self.cnt = 0

    def __iter__(self):
        return self

    def reload_buffer(self, start=-1, stop_iteration=False):
        if self.buf_cnt < self.num_read:
            if stop_iteration:
                raise StopIteration()
        if start != -1 and start != self.cnt:
            self.fp.seek(0, 0)
            self.cnt = 0
            for i in xrange(start):
                if not self.fp.readline():
                    if stop_iteration:
                        raise StopIteration
                    else:
                        break
                else:
                    self.cnt += 1
        self.buf_ind = 0
        self.buf_cnt = 0
        for line in self.fp:
            fields = line.strip().split(self.sep)
            self.datax[self.buf_cnt] = numpy.reshape(numpy.frombuffer(base64.b64decode(fields[self.field_nums[0]]), dtype=self.dtype), self.reshape_dims[0])
            self.datay[self.buf_cnt] = numpy.reshape(numpy.frombuffer(base64.b64decode(fields[self.field_nums[1]]), dtype=self.dtype), self.reshape_dims[1])
            self.buf_cnt += 1
            if self.buf_cnt >= self.num_read:
                break
        
    def next(self):
        retval = [None,None]
        if self.huge_data:
            if self.buf_ind >= self.buf_cnt:
                self.reload_buffer(stop_iteration=True, start=self.cnt)
            retval[0] = self.datax[self.buf_ind]
            retval[1] = self.datay[self.buf_ind]
            self.buf_ind += 1
        else:
            if self.cnt >= self.size:
                raise StopIteration()
            retval[0] = self.datax[self.permut[self.cnt]]
            retval[1] = self.datay[self.permut[self.cnt]]
        self.cnt += 1
        return retval

    def __getitem__(self, info):
        start, stop = -1, -1
        if isinstance(info, int):
            start,stop = info,info+1
        elif isinstance(info, slice):
            start = info.start
            stop = info.stop

        retval = [None,None]
        if start >= stop:
            return retval
        if self.huge_data:
            if start != self.cnt:
                self.reload_buffer(start=start)
            retval[0] = numpy.zeros([stop-start]+self.reshape_dims[0], dtype=self.dtype)
            retval[1] = numpy.zeros([stop-start]+([] if (self.reshape_dims[1]==1) else self.reshape_dims[1]), dtype=self.dtype)
            base_cnt = self.cnt
            for i in xrange(start,stop):
                if self.buf_ind >= self.buf_cnt:
                    self.reload_buffer(start=self.cnt)
                retval[0][self.cnt-base_cnt] = self.datax[self.buf_ind]
                retval[1][self.cnt-base_cnt] = self.datay[self.buf_ind]
                self.buf_ind += 1
                self.cnt += 1
        else:
            retval[0] = numpy.zeros([stop-start]+self.reshape_dims[0], dtype=self.dtype)
            retval[1] = numpy.zeros([stop-start]+([] if (self.reshape_dims[1]==1) else self.reshape_dims[1]), dtype=self.dtype)
            for i in xrange(start, stop):
                retval[0][i-start] = self.datax[self.permut[i]]
                retval[1][i-start] = self.datay[self.permut[i]]
        return retval

class MLPHiddenLayer(object):
    def __init__(self, rng, input, input_dim, output_dim, activation, dropout_rate=0.,W=None,b=None):
        self.input = input
        wrange = numpy.sqrt(6./(input_dim+output_dim))
        if activation == T.nnet.sigmoid:
            wrange *= 4.
        if W:
            self.W = W
        else:
            self.W = theano.shared(numpy.asarray(rng.uniform(low=-wrange, high=wrange, size=(input_dim, output_dim)), 
                                dtype=theano.config.floatX),
                                borrow=True)
        if b:
            self.b = b
        else:
            self.b = theano.shared(numpy.zeros(output_dim, dtype=theano.config.floatX), borrow=True)

        self.base_output = activation(T.dot(input, self.W)+self.b)
        self.output = self.base_output
        self.params = [self.W, self.b]

class MLPDropoutHiddenLayer(MLPHiddenLayer):
    def __init__(self, rng, input, input_dim, output_dim, activation, dropout_rate=0.):
        super(MLPDropoutHiddenLayer, self).__init__(rng=rng, input=input, input_dim=input_dim, output_dim=output_dim,
                                                    activation=activation, dropout_rate=dropout_rate)
        if dropout_rate:
            rngstr = T.shared_randomstreams.RandomStreams(rng.randint(333333))
            mask = rngstr.binomial(size=self.base_output.shape, n=1, p=(1.-dropout_rate))
            self.output = self.base_output * T.cast(mask, theano.config.floatX)
        else:
            self.output = self.base_output  

class MLPOutputLayer(object):
    def __init__(self, mlp_type, rng, input, input_dim, output_dim, activation, W=None, b=None):
        self.input = input

        #self.W = theano.shared(numpy.asarray(rng.uniform(low=-1, high=1, size=(input_dim, output_dim)), 
        #                        dtype=theano.config.floatX),
        #                        borrow=True)
        if W:
            self.W = W
        else:
            self.W = theano.shared(numpy.zeros((input_dim, output_dim), dtype=theano.config.floatX), borrow=True)

        if b:
            self.b = b
        else:
            self.b = theano.shared(numpy.zeros(output_dim, dtype=theano.config.floatX), borrow=True)

        self.ypred = (T.dot(input, self.W) + self.b) if activation is None else activation(T.dot(input, self.W)+self.b)
        if mlp_type == 'classification':
            self.y = T.argmax(self.ypred, axis=1)
        else:
            self.y = self.ypred 
        self.params = [self.W, self.b]
        self.mlp_type = mlp_type

    def cross_entropy(self, y):
        return T.sum(-y*T.log(self.y) - (1-y)*T.log(1-self.y))

    def least_squares_error(self, y):
        return T.sum((y-self.y) ** 2)
        
    def negative_log_likelihood(self, y):
        return -T.sum(T.log(self.ypred)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if self.mlp_type == 'classification':
            return T.sum(T.neq(self.y, y))
        else:
            return T.sum(abs(self.y - y))

    def get_output(self):
        return self.y

    def get_prediction(self):
        return self.ypred

class MLP(object):
    #hidden_dims: is a list of dimensions of hidden layers
    def __init__(self, mlp_type, rng, input, input_dim, hidden_dims, output_dim, hidden_activation=None, 
                    out_activation=None, dropout_rate=0.):
        self.hidden_layers = []
        self.dropout_hidden_layers = []

        prev_do_out = input
        prev_out = input
        prev_out_dim = input_dim
        for i in xrange(len(hidden_dims)):
            dhl = MLPDropoutHiddenLayer(rng=rng, input=prev_do_out, input_dim=prev_out_dim, 
                               output_dim=hidden_dims[i], activation=hidden_activation, dropout_rate=dropout_rate)
            self.dropout_hidden_layers.append(dhl)

            hl = MLPHiddenLayer(rng=rng, input=prev_out, input_dim=prev_out_dim, 
                               output_dim=hidden_dims[i], activation=hidden_activation, dropout_rate=dropout_rate,
                                W=(1.-dropout_rate)*dhl.W, b=dhl.b)
            self.hidden_layers.append(hl)

            prev_do_out = dhl.output
            prev_out = hl.output
            prev_out_dim = hidden_dims[i]

        self.do_out_layer = MLPOutputLayer(mlp_type=mlp_type, rng=rng, input=self.dropout_hidden_layers[-1].output, 
                               input_dim=hidden_dims[-1], output_dim=output_dim, activation=out_activation)

        self.out_layer = MLPOutputLayer(mlp_type=mlp_type, rng=rng, input=self.hidden_layers[-1].output, 
                               input_dim=hidden_dims[-1], output_dim=output_dim, activation=out_activation, 
                                W=(1.-dropout_rate)*self.do_out_layer.W, b=self.do_out_layer.b)

        #L1 Norm
        self.l1norm = reduce(lambda x,y: x + y, map(lambda x: abs(x.W).sum(), self.dropout_hidden_layers)) + abs(self.do_out_layer.W).sum()

        #L2 Norm Square
        self.l2norm_sq = reduce(lambda x,y: x + y, map(lambda x: (x.W ** 2).sum(), self.dropout_hidden_layers)) + (self.do_out_layer.W ** 2).sum()

        #Negative Log Likelihood
        self.negative_log_likelihood = self.do_out_layer.negative_log_likelihood;
        
        self.least_squares_error = self.do_out_layer.least_squares_error;
        self.cross_entropy = self.do_out_layer.cross_entropy;

        #Errors
        self.errors = self.out_layer.errors

        self.get_output = self.out_layer.get_output
        self.get_prediction = self.out_layer.get_prediction

        self.params = self.do_out_layer.params + reduce(lambda x,y: x + y, map(lambda x: x.params, reversed(self.dropout_hidden_layers)))


class ConvolveMaxPool(object):
    #rng: random number generator to initialize weights
    #input: matrix of dimensions input_dim
    #input_dim: tuple of (num inputs, num input feature maps, input width, input height)
    #filter_dim: tuple of (num filters, num input feature maps, filter width, filter height)
    #mp_dim: region size for max pooling
    def __init__(self, rng, input, input_dim, filter_dim, mp_dim, activation=T.tanh):
        assert filter_dim[1] == input_dim[1]
        self.input = input

        fan_in = numpy.prod(filter_dim[1:])
        fan_out = filter_dim[0] * numpy.prod(filter_dim[2:])

        #initialize weights & biases
        wrange = numpy.sqrt(6. / (fan_in+fan_out))
        self.W = theano.shared(numpy.asarray(rng.uniform(low=-wrange, high=wrange, size=filter_dim), 
                                                dtype=theano.config.floatX),
                                borrow=True)
        self.b = theano.shared(numpy.zeros(filter_dim[0], dtype=theano.config.floatX), borrow=True)

        #convolve
        self.cout = T.nnet.conv.conv2d(input=input, filters=self.W, filter_shape=filter_dim, image_shape=input_dim)

        #max pooling
        if mp_dim:
            self.mpout = downsample.max_pool_2d(input=self.cout, ds=mp_dim, ignore_border=True)
        else:
            self.mpout = self.cout

        self.output = activation(self.mpout + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]

class CNN(object):
    #input_dim: tuple of (num inputs, num input feature maps, input width, input height)
    #filter_dims: list of tuples of (num filters, num input feature maps, filter width, filter height)
    #mp_dims: list of region sizes for max pooling
    #mlp_hidden_dims: list of dimensions of hidden layers for MLP
    def __init__(self, cnn_type, rng, input, input_dim, filter_dims, mp_dims, mlp_hidden_dims, mlp_output_dim,
                    conv_activation, mlp_hidden_activation, mlp_out_activation, dropout_rate=0.):
        assert len(filter_dims) == len(mp_dims)

        #create convolution and max pooling layers
        self.input = input
        self.conv_layers = []
       
        prev_out = input
        prev_dim = input_dim
        
        for i in xrange(len(filter_dims)):
            cl = ConvolveMaxPool(rng=rng, input=prev_out, input_dim=prev_dim, filter_dim=filter_dims[i], 
                                    mp_dim=mp_dims[i], activation=get_activation(conv_activation))
            prev_out = cl.output
            prev_dim = (prev_dim[0], 
                        filter_dims[i][0], 
                        (prev_dim[2] - filter_dims[i][2] + 1)/mp_dims[i][0] if mp_dims[i] else (prev_dim[2]-filter_dims[i][2]+1), 
                        (prev_dim[3] - filter_dims[i][3] + 1)/mp_dims[i][1] if mp_dims[i] else (prev_dim[3]-filter_dims[i][3]+1))
            self.conv_layers.append(cl)

        #create MLP
        mlp_input = prev_out.flatten(2)
        mlp_input_dim = numpy.prod(prev_dim[1:])
        self.mlp = MLP(cnn_type, rng, mlp_input, mlp_input_dim, mlp_hidden_dims, mlp_output_dim, hidden_activation=get_activation(mlp_hidden_activation), 
                        out_activation=get_activation(mlp_out_activation),dropout_rate=dropout_rate)

        self.l1norm = self.mlp.l1norm + (reduce(lambda x,y: x + y, map(lambda x: abs(x.W).sum(), self.conv_layers)) if self.conv_layers else 0)
        self.l2norm_sq = self.mlp.l2norm_sq + (reduce(lambda x,y: x + y, map(lambda x: (x.W ** 2).sum(), self.conv_layers)) if self.conv_layers else 0)
        self.negative_log_likelihood = self.mlp.negative_log_likelihood
        self.least_squares_error = self.mlp.least_squares_error
        self.cross_entropy = self.mlp.cross_entropy;
        self.errors = self.mlp.errors
        self.params = self.mlp.params + (reduce(lambda x, y: x + y, map(lambda x: x.params, reversed(self.conv_layers))) if self.conv_layers else [])
        self.get_output = self.mlp.get_output
        self.get_prediction = self.mlp.get_prediction
