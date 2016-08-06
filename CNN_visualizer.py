'''
kurma Utils:First Layer Activation Visualizer
Currently runs the first layer filters on top of input images
Note: Add the utils/modules/src file to your sys path
'''


import sys
import os
from modules.utils.fsutils import DataFile
from matplotlib import pyplot as plt
import cv2
import pickle as pkl
import theano as T
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K
import numpy as np



class LayerVisualizer:

	configs = None
	params = None
	conv_filter_dims = 0
	mlp_filter_dims = 0
	def __init__(self, model_path):
		# try:
		os.path.isfile(model_path)
		print model_path
		mf = DataFile(model_path, 'r', None, is_binary=True).get_fp()
		model_info = pkl.load(mf)
		self.configs = model_info['config']
		self.params =  model_info['params']
		self.conv_filter_dims = self.configs['filter_dims']
		self.mlp_filter_dims = self.configs['mlp_hidden_dims']
		self.nb_input_features = self.configs['input_features']
		self.img_width = self.configs['input_wd']
		self.img_height = self.configs['input_ht']
			#Building Keras model
			# model = Sequential()
			# model.add(ZeroPadding2D(1,1), batch_input_shape = (1, nb_input_features,img_width, img_height)
			# model.add()	
		# except:
		# 	raise ValueError("No file found")
		# 	sys.exit(-1)	
		

	
	def get_filters(self, layer=1):
		weights =  self.params[-2 * layer]
		shape = weights.shape
		weights_t = weights.transpose(3,2,1,0)
		print weights_t.shape
		self.filters = [weights_t[:,:,:,i] for i in xrange(shape[0])]
		print len(self.filters)
		return self.filters

	def visualize(self, input_img):
		in_shape = input_img.shape
		if len(in_shape) == 2:
			in_channels = 1
		elif len(in_shape) == 3:
			in_channels = 3
			 
		if in_channels != self.nb_input_features:
			print('No. of channels do not match')
			sys.exit(-1) 	 
		if in_shape[0] != self.img_width or in_shape[1] != self.img_height:
			print('Image Dimensions not matching to network')
			input_img = cv2.resize(input_img, (self.img_width, self.img_height))
		out_imgs=[]	
		# if self.nb_input_features == 1:
		# 	for i in self.filters:
		# 		out_imgs.append(cv2.filter2D(input_img, -1, i[:,:,0]))
		# if self.nb_input_features == 3:
		for i in self.filters:
			#i = i/i.sum()
			channels = cv2.split(input_img)
			#channels = [i/i.sum() for i in channels]
			filters_ch = cv2.split(i)
			out = cv2.merge([ cv2.filter2D(l,-1,m) for l,m in zip(channels, filters_ch) ])
			out_imgs.append(out)
		return out_imgs


def usage():
	print('Usage: python CNN_visualizer <path>')
	return 

if __name__ == '__main__':

	if len(sys.argv) < 2:
		usage()
		sys.exit(-1)	
	a = LayerVisualizer(sys.argv[1])
	#img = cv2.imread(sys.argv[2],0)
	filters = a.get_filters(1)
	#do the image processing part here.
	#TODO: Add baseImageProc support
#	img1 = (img - img.mean())/img.std()
#	output_imgs = a.visualize(img1)
#	output_imgs = [ cv2.convertScaleAbs((i - i.min())/i.ptp(),0,255) for i in output_imgs]
	cnt = 0
	for i in filters:
                m = cv2.resize(i, (i.shape[0]*11, i.shape[1]*11))
                m = np.uint8(255*(m-m.min() / m.ptp()))
		#plt.imsave(str(cnt)+'.jpg', m, cmap='jet')
                cv2.imwrite(str(cnt)+'.jpg',m)
                cnt+=1
	
