''' dummy.py: An abstract class for extracting patches of specific types'''

__author__="Ameya Joshi"
__copyright__="Copyright 2015, SigTuple Technologies Pvt Ltd"
__version__="0.1"
__maintainer__="Rohit Kumar Pandey"
__email__="ameya@sigtuple.com"
__status__="development"

import numpy as np
import cv2
import sys
import os
from scipy import ndimage
import math
from cell import Cell
from modules.inputreaders import baseinputreader

class Dummy:

	def __init__(self, config, label, attrib_flag, logger):
		self.config = config
		self.label = label
		self.logger = logger
		self.attrib_flag = attrib_flag
		self.curr_img = None
		self.curr_img_name = None

	def extract_patches(self, input_info, all_global_attributes = False):
		if all_global_attrib and num_patches <= 0:
            return ({}, {})
        input_reader = baseinputreader.factory(input_info['type'], self.logger, input_info['path'], in_ppm=[input_info['ppm_x'], input_info['ppm_y']], out_ppm=self.out_ppm)
        input_reader.reset(jump=self.input_stride_size, patch_size=self.input_patch_size, magn=self.input_magn)    
        out_patches = {}
        out_attribs = {'name':{}}
        for (x,y,img) in input_reader:
        	img_patches={}
        	self.curr_img = img
        	self.curr_img_name = input_reader.get_name()
        	if None is img:
        		self.logger.debug('image %s not found'%self.curr_img_name)
        		continue
        	img_patches[self.curr_img_name.split('.')[0]] = img
        	self.logger.debug('The img_name is %s'%"_".join([self.curr_img_name.split('.')[0],'xray',str(0),str(0)]))	
		    out_patches.update(img_patches)
		    out_attribs["name"]=self.curr_img_name.split('.')[0]
		    return (out_patches, out_attribs)