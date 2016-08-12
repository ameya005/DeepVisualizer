#!/usr/bin/env python

import os,sys
import numpy
import base64
import json
import time
import pickle
import random
import signal
import weakref
from datetime import datetime
from bson.objectid import ObjectId
import theano
import theano.tensor as T
from cnnutils import *
from mdd import MDD

class CNNTrainer(object):
    instances = []

    @staticmethod
    def save_and_exit_handler(signum, frame):
        if not CNNTrainer.instances:
            return
        CNNTrainer.instances[0].SAVE_AND_EXIT = True
   
    @staticmethod
    def double_l2reg_handler(signum, frame):
        if not CNNTrainer.instances:
            return
        CNNTrainer.instances[0].DOUBLE_L2REG = True
   
    @staticmethod
    def halve_lr_handler(signum, frame):
        if not CNNTrainer.instances:
            return
        CNNTrainer.instances[0].HALVE_LR = True
 
    def __init__(self, sys_config, module_config, config, logger):

        self.sys_config = sys_config
        self.module_config = module_config
        self.config = config
        self.db_config = self.sys_config.get('db', {})
        self.logger = logger
        self.SAVE_AND_EXIT = False
        self.DOUBLE_L2REG = False
        self.HALVE_LR = False

        #load global params
        self.train_data_base_path = self.sys_config['paths'].get('train_data', '')
        if not self.train_data_base_path or not os.path.exists(self.train_data_base_path):
            raise Exception('training data base path %s does not exist' % self.train_data_base_path)
        self.model_repo_base_path = self.sys_config['paths'].get('model_repo', '')
        if not self.model_repo_base_path or not os.path.exists(self.model_repo_base_path):
            raise Exception('model repository base path %s does not exist' % self.model_repo_base_path)

        #load db config
        self.dbclient = MDD(self.db_config, self.logger)
        if not self.dbclient:
            raise Exception('could not initialise db connection')
        self.db = self.db_config['database']
        self.train_data_table = self.db_config['tables']['train_data']
        self.model_version_table = self.db_config['tables']['model_version']
        self.model_table = self.db_config['tables']['models']
        self.sync_table = self.db_config['tables']['sync']

        self.logger.info('loading config parameters')
        self.model_id = self.config.get('model_id', None)
        if not self.model_id:
            raise Exception('no model id provided for training in config')

        self.train_data_id = self.config.get('train_data_id', None)
        if not self.train_data_id:
            raise Exception('no training data id provided for training')

        self.huge_data = self.config.get('huge_data', False)
        self.incr = self.config.get('incr', False)
        self.data_iters = {}
        self.batch_data = {}

        if self.incr:
            self.load_model_config()

        self.no_param_change = self.config.get('no_param_change', False)
        self.batch_size = self.config.get('batch_size')
        self.epochs = self.config.get('epochs')
        self.input_xy_indexes = self.config.get('input_xy_indexes', [0,1])
        self.input_delim = self.config.get('input_separator', '|')
        self.input_wd = self.config.get('input_wd')
        self.input_ht = self.config.get('input_ht')
        self.input_features = self.config.get('input_features', 1)
        self.input_dim = (self.batch_size, self.input_features, self.input_wd, self.input_ht)
        self.filter_dims = self.config.get('filter_dims')
        self.mp_dims = self.config.get('mp_dims')
        self.mlp_hidden_dims = self.config.get('mlp_hidden_dims')
        self.mlp_output_dim = numpy.prod(self.config.get('mlp_output_dim'))
        self.output_dim = numpy.prod(self.config.get('output_dim'))
        self.conv_activation = self.config.get('conv_activation')
        self.mlp_hidden_activation = self.config.get('mlp_hidden_activation')
        self.mlp_out_activation = self.config.get('mlp_out_activation')
        self.cnn_type = self.config.get('type')
        self.mu = self.config.get('momentum', 0.)
        self.dropout_rate = self.config.get('dropout_rate', 0.)
        self.anneal_epochs = self.config.get('anneal_epochs', 25)
        self.max_buffer_data = self.module_config['train'].get('max_buffer_data', 1)
        self.logger.info('config parameters loaded') 
  
        self.logger.info('setting up signal handlers for training')
        signal.signal(signal.SIGINT, CNNTrainer.save_and_exit_handler)
        signal.signal(signal.SIGUSR1, CNNTrainer.halve_lr_handler)
        signal.signal(signal.SIGUSR2, CNNTrainer.double_l2reg_handler)

        #add to static list
        self.__class__.instances.append(weakref.proxy(self))


    def get_next_model_version(self):
        max_retry_cnt = 5
        retry_cnt = 0

        self.model_version = None
        while(retry_cnt < max_retry_cnt):
            #fetch max version as of now
            (mdoc, status) = self.dbclient.get_one(self.db, self.model_version_table, {'model_id': self.model_id}, sort=[('version_id', -1)])
            if not mdoc or status:
                #first model version
                mdoc = {'version_id': -1}
            new_version = '%s.%s.%s.%s.%d' % (self.model_info['field'], self.model_info['product'], self.model_info['group'], self.model_info['id'], mdoc['version_id']+1)
            #try to update into DB
            if "model" in self.sys_config['sync_entities']:
                sync_status ='sync_pending:%s' % self.sys_config.get('instance_id', '')
            else:
                sync_status = 'not_synced'
            (self.model_control_id, status) = self.dbclient.post_data(self.db, self.model_version_table, {'instance_id': self.sys_config.get('instance_id', ''), 'version_id': mdoc['version_id']+1, 'version': new_version, 'model_id': self.model_id, 'train_data_id': self.train_data_id, 'status': 'in process', 'ts': datetime.utcnow(), 'sync_status': sync_status})
            if not status:
                self.model_version = new_version
                self.logger.info('version id %s set for current model, updated in db' % self.model_version)
                break
            retry_cnt += 1
        if not self.model_version:
            raise Exception('could not register new version for model %s in db' % (self.model_id))

    def load_model_config(self):
        in_model_version = self.config.get('input_model_version', None)
        #get model path from DB
        (mdocs, status) = self.dbclient.get_data(self.db, self.model_version_table, {'version': in_model_version})
        if status:
            raise Exception('could not fetch model version info from db for input version %s' % in_model_version)

        if mdocs[0]['model_id'] != self.model_id:
            raise Exception('input version model id %s and model id %s provided in config do not match' % (mdocs[0]['model_id'], self.model_id))

        model_status = mdocs[0].get('status', '')
        model_sync_status = mdocs[0].get('sync_status', '')
        model_instance_id = mdocs[0].get('instance_id', '')
        if model_status != 'success' or  ( model_sync_status not in ['success', 'sync_pending:%s' % self.sys_config.get('instance_id', '')] and model_instance_id != self.sys_config.get('instance_id', '')):
            raise Exception('cannot use input model version %s with status %s and sync_status %s' % (in_model_version, model_status, model_sync_status))

        in_model_path = mdocs[0]['path']

        if not in_model_path or not os.path.exists(in_model_path):
            raise Exception('could not load input model data for incremental training')
        self.logger.info('loading input model parameters from model file %s' % in_model_path)
        mf = DataFile(in_model_path, 'r', self.logger, is_binary=True).get_fp()
        self.initial_model_info = pickle.load(mf)
        mf.close()
        mconf = self.initial_model_info['config']
        if 'epochs' in self.config:
            mconf['epochs'] = self.config.get('epochs')
        if 'batch_size' in self.config:
            mconf['batch_size'] = self.config.get('batch_size')
        if 'learning_rate' in self.config:
            mconf['learning_rate'] = self.config.get('learning_rate')
        if 'l2reg' in self.config:
            mconf['l2reg'] = self.config.get('l2reg')
        if 'momentum' in self.config:
            mconf['momentum'] = self.config.get('momentum')
        if 'dropout_rate' in self.config:
            mconf['dropout_rate'] = self.config.get('dropout_rate')
        if 'anneal_epochs' in self.config:
            mconf['anneal_epochs'] = self.config.get('anneal_epochs')
        mconf['input_model_path'] = self.config.get('input_model_path')
        self.config = mconf
        self.logger.info('loaded model config for input version %s' % in_model_version)

    def load_batch_data(self, dset, start, stop):
        [x,y] = self.data_iters[dset][start:stop]
        if self.cnn_type == 'classification':
            y = numpy.array(y, dtype=numpy.int32)
        else:
            y = numpy.array(y, dtype=numpy.float32)
        x = numpy.array(x, dtype=numpy.float32)
        if dset not in self.batch_data:
            self.batch_data[dset] = [theano.shared(x, borrow=True), theano.shared(y, borrow=True)]
        else:
            self.batch_data[dset][0].set_value(x)
            self.batch_data[dset][1].set_value(y)
   
    def load_data(self):
        #fetch input data dir
        (mdocs, status) = self.dbclient.get_data(self.db, self.train_data_table, {'_id': ObjectId(self.train_data_id)})
        if status:
            raise Exception('could not fetch train data info from db for id %s' % str(self.train_data_id))
        if mdocs[0]['model_id'] != self.model_id:
            raise Exception('model id from config %s does not match train data model id %s' % (self.model_id, mdocs[0]['model_id']))
        input_status = mdocs[0].get('status', '')
        sync_status = mdocs[0].get('sync_status','')
        instance_id = mdocs[0].get('instance_id',None)
        if input_status != "success" or ( sync_status not in ['success', 'sync_pending:%s' % self.sys_config.get('instance_id', '')] and instance_id != self.sys_config.get('instance_id', '')):
            raise Exception('cannot use training data %s with status %s and sync_status %s for training' % (str(self.train_data_id), input_status, sync_status))
        self.logger.info('loading training data from %s' % mdocs[0]['path'])
        self.train_data_path = os.path.join(mdocs[0]['path'], self.module_config['train'].get('train_file'))
        self.validate_data_path = os.path.join(mdocs[0]['path'], self.module_config['train'].get('validate_file'))
        self.test_data_path = os.path.join(mdocs[0]['path'], self.module_config['train'].get('test_file'))
        self.data_counts = mdocs[0]['counts']

        #create iters
        self.logger.info('creating train data iterator from file %s' % self.train_data_path)
        self.data_iters['train'] = CNNDataFileIter(self.logger, self.train_data_path, [list(self.input_dim[1:]), self.output_dim], huge_data=self.huge_data, field_nums=self.input_xy_indexes, sep=self.input_delim, size=self.data_counts['train'], max_buffer_data=self.max_buffer_data)
        self.logger.info('creating validate data iterator from file %s' % self.validate_data_path)
        self.data_iters['validate'] = CNNDataFileIter(self.logger, self.validate_data_path, [list(self.input_dim[1:]), self.output_dim], huge_data=self.huge_data, field_nums=self.input_xy_indexes, sep=self.input_delim, size=self.data_counts['validate'], max_buffer_data=self.max_buffer_data)
        self.logger.info('creating test data iterator from file %s' % self.test_data_path)
        self.data_iters['test'] = CNNDataFileIter(self.logger, self.test_data_path, [list(self.input_dim[1:]), self.output_dim], huge_data=self.huge_data, field_nums=self.input_xy_indexes, sep=self.input_delim, size=self.data_counts['test'], max_buffer_data=self.max_buffer_data)
        #create required theano vars
        self.load_batch_data('train', 0, self.batch_size)
        self.load_batch_data('validate', 0, self.batch_size)
        self.load_batch_data('test', 0, self.batch_size)
        self.logger.info('Creating variables of batch size %d', self.batch_size)
    def train(self):
        rng = numpy.random.RandomState()
   
        self.logger.info('loading data files')
        self.load_data()
        self.logger.info('loading data done')

        num_train_data = self.data_iters['train'].get_size()
        num_validation_data = self.data_iters['validate'].get_size()
        num_test_data = self.data_iters['test'].get_size()

        num_train_runs = num_train_data/self.batch_size 
        num_validation_runs = num_validation_data/self.batch_size
        num_test_runs = num_test_data/self.batch_size


        #create new model version number
        #load model info
        (mdocs, status) = self.dbclient.get_data(self.db, self.model_table, {'id': self.model_id})
        if status:
            raise Exception('could not fetch info for model %s from db' % (self.model_id))
        self.model_info = mdocs[0]
        self.get_next_model_version()

        try:
            #symbolic variables
            index = T.lscalar()
            x = T.tensor4('x')
            y = None
            if self.cnn_type == 'classification':
                y = T.ivector('y')
            else:
                y = T.matrix('y')
    
            self.learning_rate = theano.shared(numpy.cast['float32'](self.config.get('learning_rate', 0.01)))
            self.l2reg = theano.shared(numpy.cast['float32'](self.config.get('l2reg', 0.01)))
    
            self.logger.info('initial learning rate: %f, initial l2reg: %f' % (self.learning_rate.get_value(), self.l2reg.get_value()))
        
            #create cnn
            self.logger.info('creating CNN')
            cnn = CNN(cnn_type=self.cnn_type, rng=rng, input=x, input_dim=self.input_dim, filter_dims=self.filter_dims, 
                        mp_dims=self.mp_dims, mlp_hidden_dims=self.mlp_hidden_dims, mlp_output_dim=self.mlp_output_dim,
                        conv_activation=self.conv_activation, mlp_hidden_activation=self.mlp_hidden_activation, 
                        mlp_out_activation=self.mlp_out_activation, dropout_rate=self.dropout_rate)
            self.logger.info('CNN created')
        
            if self.incr:
                self.logger.info('initializing CNN with input model parameters')
                set_model_params(cnn, self.initial_model_info['params'])
       
            self.logger.info('creating theano symbolic variables and functions')
            velocities = [theano.shared(param.get_value()*0., broadcastable=param.broadcastable) for param in cnn.params]
        
            cost =  get_cost_function(self.config.get('cost_func', None), cnn, y) + self.l2reg*cnn.l2norm_sq
        
            grads = T.grad(cost, cnn.params)
        
            updates = None
            if self.mu == 0.0:
                updates = [(param, param - self.learning_rate*grad) for (param,grad) in zip(cnn.params, grads)]
            else:
                updates = [(param, param - self.learning_rate*grad + self.mu*velocity) for (param, grad, velocity) in zip(cnn.params, grads, velocities)]+\
                        [(velocity, self.mu*velocity - self.learning_rate*grad) for (velocity, grad) in zip(velocities, grads)]
            
            train_model = theano.function(inputs=[], outputs=cost, updates=updates, 
                                givens= {
                                    x: self.batch_data['train'][0],
                                    y: self.batch_data['train'][1] 
                                })
        
            validate_model = theano.function(inputs=[], outputs=cnn.errors(y),  
                                givens= {
                                    x: self.batch_data['validate'][0],
                                    y: self.batch_data['validate'][1]
                                })
            test_model = theano.function(inputs=[], outputs=cnn.errors(y),
                                givens= {
                                    x: self.batch_data['test'][0],
                                    y: self.batch_data['test'][1]
                                })
            self.logger.info('theano configuration done')
        
            # early-stopping parameters
            patience = max(10,self.epochs/2)*num_train_runs  # look as this many batches regardless
            patience_increase = 2  # wait this much longer when a new best is
                                   # found
            improvement_threshold = 0.99995  # a relative improvement of this much is
                                           # considered significant
            best_model_params = None
            best_validation_error = numpy.inf
            best_iter = 0
            best_test_error = numpy.inf
            best_model_error = numpy.inf
            start_time = time.time()
            temp_time = start_time
            epoch_cost = 0
            prev_epoch_cost = 0
            num_bad_epochs = 0
            num_good_epochs = 0
            num_bad_overfits = 0
            max_good_epochs = self.config.get('max_good_epochs', 20) #change learning rate when this hits
            max_bad_epochs = self.config.get('max_bad_epochs', 3) #change learning rate when this hits
            max_bad_overfits = self.config.get('max_bad_overfits', 5) #change l2reg when this hits
            prev_validation_error = numpy.inf
            prev_epoch_cost = numpy.inf
        
            if self.incr:
                best_validation_error = self.config.get('validation_error', numpy.inf)
                best_test_error = self.config.get('test_error', numpy.inf)
                best_model_error = self.config.get('model_error', numpy.inf)
                if best_model_error == numpy.inf:
                    best_model_error = (best_validation_error*num_validation_data+best_test_error*num_test_data)/(1.0*(num_validation_data+num_test_data))
                best_model_params = self.initial_model_info['params']
        
            epoch = 0
            done_looping = False
        
            self.logger.info('starting training')
            while (epoch < self.epochs) and not done_looping and not self.SAVE_AND_EXIT:
                epoch_cost = 0
        
                #double l2reg or halve LR if signal received
                if self.DOUBLE_L2REG or self.HALVE_LR:
                    num_bad_epochs = 0
                    num_good_epochs = 0
                    num_bad_overfits = 0
                    if self.DOUBLE_L2REG:
                        new_l2reg = self.l2reg.get_value()*2.
                        self.logger.info('signal received: changing l2reg from %g to %g' % (self.l2reg.get_value(), new_l2reg))
                        self.l2reg.set_value(new_l2reg)
                        self.DOUBLE_L2REG = False
                    if self.HALVE_LR:
                        new_lr = self.learning_rate.get_value()*0.5
                        self.logger.info('signal received: changing learning rate from %g to %g' % (self.learning_rate.get_value(), new_lr))
                        self.learning_rate.set_value(new_lr)
                        self.HALVE_LR = False
        
                #shuffle training
                self.data_iters['train'].shuffle_and_reset()
                
                #anneal learning rate
                if epoch > 0 and epoch % self.anneal_epochs == 0:
                    self.learning_rate.set_value(numpy.cast['float32'](self.learning_rate.get_value()*0.5))
                    self.logger.info('annealing learning rate, new value: %g' % (self.learning_rate.get_value()))
        
                for batch_index in xrange(num_train_runs):
                    itr = epoch * num_train_runs + batch_index
        
                    #get new training batch
                    self.load_batch_data('train', batch_index*self.batch_size, (batch_index+1)*self.batch_size)
        
                    epoch_cost += train_model()
        
                    if (itr + 1) % num_train_runs == 0:
                        validation_losses = []
                        for i in xrange(num_validation_runs):
                            self.load_batch_data('validate', i*self.batch_size, (i+1)*self.batch_size)
                            validation_losses.append(validate_model())
                        validation_error = (numpy.sum(validation_losses)*1.0)/(num_validation_runs*self.batch_size)
                        
                        self.logger.info(
                            'epoch %i, time_taken %f, batch %i/%i, epoch_cost %f, validation error %f %%' %
                            (
                                epoch,
                                (time.time() - temp_time),
                                batch_index + 1,
                                num_train_runs,
                                epoch_cost,
                                validation_error * 100.
                            )
                        )
                        temp_time = time.time()

                        # test it on the test set
                        test_losses = []
                        for i in xrange(num_test_runs):
                            self.load_batch_data('test', i*self.batch_size, (i+1)*self.batch_size)
                            test_losses.append(test_model())
                        test_error = (numpy.sum(test_losses)*1.0)/(num_test_runs*self.batch_size)
        
                        # if we got the best validation score until now
                        model_error = (validation_error*num_validation_data+test_error*num_test_data)/(1.0*(num_validation_data+num_test_data))
                        self.logger.info('Model Error: %f, test_error: %f', model_error*100, test_error * 100)
                        if model_error < best_model_error:
                            #improve patience if loss improvement is good enough
                            if model_error < best_model_error * improvement_threshold:
                                patience = max(patience, itr * patience_increase)
                            # save best model
                            best_validation_error = validation_error
                            best_iter = itr
                            best_test_error = test_error
                            best_model_error = model_error
                            best_model_params = get_model_params(cnn)
        
                            self.logger.info(('     epoch %i, test error of '
                                   'best model %f %%, model error %f %%') %
                                  (epoch, best_test_error * 100., best_model_error*100.))
                            num_bad_epochs = 0
                            num_bad_overfits = 0
                            if epoch_cost < prev_epoch_cost:
                                num_good_epochs += 1
                            #if not self.no_param_change and num_good_epochs >= max_good_epochs:
                            #    new_learning_rate = 2.*self.learning_rate.get_value()
                            #    self.logger.info('changing learning rate from %g to %g' % (self.learning_rate.get_value(), new_learning_rate))
                            #    self.learning_rate.set_value(new_learning_rate)
                            #    num_good_epochs = 0
                            #    num_bad_epochs = 0
                        else:
                            if epoch_cost > prev_epoch_cost:
                                num_bad_epochs += 1
                                num_good_epochs = 0
                            if validation_error > prev_validation_error and \
                                    epoch_cost < prev_epoch_cost:
                                num_bad_overfits += 1
                            if not self.no_param_change:
                                if num_bad_epochs >= max_bad_epochs:
                                    new_learning_rate = self.learning_rate.get_value()/2.
                                    self.logger.info('changing learning rate from %g to %g' % (self.learning_rate.get_value(), new_learning_rate))
                                    self.learning_rate.set_value(new_learning_rate)
                                    num_bad_epochs = 0
                                    num_bad_overfits = 0
                                if num_bad_overfits >= max_bad_overfits:
                                    new_l2reg = self.l2reg.get_value()*2.
                                    self.logger.info('changing l2reg from %g to %g' % (self.l2reg.get_value(), new_l2reg))
                                    self.l2reg.set_value(new_l2reg)
                                    num_bad_overfits = 0
                                    num_bad_epochs = 0
                        prev_validation_error = validation_error
                        prev_epoch_cost = epoch_cost 
                    if patience <= itr:
                        done_looping = True
                        break
                epoch += 1
       
            total_time_taken = time.time() - start_time
            self.logger.info(('training complete, total time taken %f. Best model has validation score of %f %% '
                   'obtained at iteration %i, with test performance %f %%') %
                  (total_time_taken, best_validation_error * 100., best_iter + 1, best_test_error * 100.))

            model_stats = {'time_taken': total_time_taken, 'model_error': best_model_error * 100, 'validation_error': best_validation_error * 100, 'test_error': best_test_error * 100}
            #save best model
            self.output_model_path = os.path.join(self.model_repo_base_path, self.model_id, self.model_version)
            self.logger.info('saving best model at %s' % (self.output_model_path))
            if not os.path.exists(os.path.join(self.model_repo_base_path, self.model_id)):
                os.makedirs(os.path.join(self.model_repo_base_path, self.model_id))
            self.config['validation_error'] = best_validation_error
            self.config['test_error'] = best_test_error
            self.config['model_error'] = best_model_error
            outf = DataFile(self.output_model_path, 'w', self.logger, is_binary=True).get_fp()
            pickle.dump({'config': self.config, 'params': best_model_params}, outf)
            outf.close()
            #update stats in db
            self.logger.info('model file written, updating stats in db')
            (num_updated, status) = self.dbclient.update_data(self.db, self.model_version_table, {'version': self.model_version}, {'$set': {'status':'success', 'ts': datetime.utcnow(), 'stats': model_stats, 'path': self.output_model_path}})
            if status:
                raise Exception('could not update success status in db for model version %s in version control table' % (self.model_version))
            if "model" in self.sys_config['sync_entities']:
                self.logger.info('scheduling model for sync')
                sync_record = {'instance_id': self.sys_config.get('instance_id', ''), 'sync_type': 'model_version', 'sync_paths': [self.output_model_path], 'status': 'pending', 'source_table': self.model_version_table, 'source_key': str(self.model_control_id), 'ts': datetime.utcnow()}
                (syncid, status) = self.dbclient.post_data(self.db, self.sync_table, sync_record)
                if status:
                    raise Exception('could not schedule sync of model version id %s' % str(self.model_control_id))
                self.logger.info('sync for model version %s scheduled successfully' % str(self.model_control_id))
            return self.model_version
        except Exception as e:
            #update failure status in db
            self.logger.exception('caught exception during training, updating failure status in db')
            (num_updated, status) = self.dbclient.update_data(self.db, self.model_version_table, {'version': self.model_version}, {'$set': {'status': 'failure', 'ts': datetime.utcnow()}})
            if status:
                self.logger.error('could not update failure status in db for model version %s in version control table' % (self.model_version))
            raise e

    def cleanup(self):
        pass
