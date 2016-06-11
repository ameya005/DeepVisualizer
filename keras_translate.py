'''
Translating current theano models directly to Keras
'''

import theano as T
import json
import cv2
import os,sys
import pickle as pkl
import numpy as np
from modules.utils.fsutils import DataFile
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import backend as K
import time

class KerasTranslator:

	def __init__(self, model_path):
		if model_path.split('.')[-1] == 'json':
			print('JSON import not supported yet')
			sys.exit(-1)
		else:
			mf = DataFile(model_path,'r',None,is_binary=True).get_fp()
			model_info = pkl.load(mf)
			self.configs = model_info['config']
			print self.configs
			self.params =  model_info['params']
			self.conv_filter_dims = self.configs['filter_dims']
			
			self.mlp_filter_dims = self.configs['mlp_hidden_dims']
			self.nb_input_features = self.configs['input_features']
			self.img_width = self.configs['input_wd']
			self.img_height = self.configs['input_ht']
			self.conv_activation=self.configs['conv_activation']
			self.mlp_hidden_activation =self.configs['mlp_hidden_activation']
			self.mp_dims = self.configs['mp_dims']
			#Building a keras model
			self.model = Sequential()
			self.model.add(ZeroPadding2D((1,1), batch_input_shape = (1,self.nb_input_features, self.img_width, self.img_height)))
			first_layer = self.model.layers[-1]
			self.input_img = first_layer.input 
			#Placeholder for input img
			input_img = first_layer.input

			for i,j in zip(self.conv_filter_dims, self.mp_dims):
				#print i
				self.model.add(Convolution2D(i[0],i[2],i[3],activation=self.conv_activation,name=('conv_%d', self.conv_filter_dims.index(i))))
				self.model.add(MaxPooling2D(tuple(j)))
			
			#loading weights of trained model
			layer_weights = []
			for i in xrange(0,len(self.conv_filter_dims)):
				#print('Hi %d %d' % ((-2 * (i+1)), (2*i+1)) )
				# for j in xrange(self.conv_filter_dims[i][0])
				# 	layer_weights.append(self.params[-2 * (i)][]	
				layer_weights = list([self.params[-2* (i+1)], self.params[-2*(i+1) + 1] ])
				#ll = [layer_weights[j,:,:,:] for j in xrange(layer_weights.shape[0])]
				wh = self.model.layers[2*i+1].get_weights()

				#print len(ll), len(wh), ll[0].shape, wh[0].shape
				print(self.model.layers[2*i + 1])
				#print layer_weights	
				print self.model.layers[2*i+ 1 ].nb_filter, self.model.layers[2*i+1].nb_row, self.model.layers[2*i+1].nb_col

				self.model.layers[2*i + 1].set_weights(layer_weights)
				#print i
			print 'Model loaded'
		# util function to convert a tensor into a valid image
	def deprocess_img(self,x, num_channels):
	    # normalize tensor: center on 0., ensure std is 0.1
	    x -= x.mean()
	    x /= (x.std() + 1e-5)
	    x *= 0.1

	    # clip to [0, 1]
	    x += 0.5
	    x = np.clip(x, 0, 1)

	    # convert to RGB array
	    x *= 255
	    print x.shape
	    
	    x = x.transpose(2,3,1,0)	
	    x = np.clip(x, 0, 255).astype('uint8')
	    return x

	def normalize(self,x):
	    # utility function to normalize a tensor by its L2 norm
	    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

	def visualize(self, layer_num, img_data=None, choose_img=False):
		print 'Processing layer %d'%(layer_num)	
		kept_filters=[]		
		for filter_idx in range(0, self.model.layers[2*layer_num - 1].output_shape[1]):
			print 'Processing filter %d'%(filter_idx)
			start_time = time.time()

			#Loss function to maximize the activation
			layer_output = self.model.layers[2*layer_num-1].output
			loss = K.mean( layer_output[:,filter_idx,: ,:])

			#Computing grads
			grads = K.gradients(loss, self.input_img)[0]
			grads = self.normalize(grads)

			iterate = K.function([self.input_img], [loss, grads])
			step = 1
			if choose_img is False:
				input_img_data = np.random.random((1,self.nb_input_features, self.img_width, self.img_height)) * 20 + 128
				print 'img_data :',input_img_data.shape
			else:
				if len(img_data.shape) == 2:
					tmp = np.expand_dims(img_data, axis=2)
					tmp = np.expand_dims(tmp, axis = 3)
					print tmp.shape
				else:
					tmp = np.expand_dims(img_data, axis=3)

				input_img_data = np.float32(tmp.transpose(3, 2, 0, 1)) 
				print 'img_data:',input_img_data.shape	

			for i in range(20):
				loss_value, grads_value = iterate([input_img_data])
				input_img_data += grads_value * step

				print('Current loss value:%f'%(loss_value))
				# if loss_value <= 0:
				# 	break

			
			img = self.deprocess_img(input_img_data, self.nb_input_features)
			kept_filters.append((img, loss_value ))

			end_time=time.time()
			print('Filtered img %d processed in %ds' % (filter_idx, end_time - start_time))			
		return kept_filters


if __name__ == '__main__':
	
	k = KerasTranslator(sys.argv[1])
	if len(sys.argv) > 4:
		img1 = cv2.imread(sys.argv[4],0)
		filters = k.visualize(int(sys.argv[2]), img_data=img1, choose_img=True)
	else:
		filters = k.visualize(int(sys.argv[2]))
	output_dir = '.'
	if len(sys.argv) > 3 :
		output_dir = str(sys.argv[3])
		
	cnt = 0
	for i in filters:
		print i[1]
		#new = i[0].transpose(2,3,0,1)

		new = i[0][:,:,:,0]
		#cv2.imshow('img', i[0])
		#cv2.waitKey(-1)
		cv2.imwrite(output_dir+'/'+str(cnt)+'.jpg', new)
		cnt+=1			


