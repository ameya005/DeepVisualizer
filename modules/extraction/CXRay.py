''' CXRay.py: An abstract class for extracting patches from Chest XRays'''

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
from scipy.signal import argrelmin, argrelmax
import math
from cell import Cell
from modules.inputreaders import baseinputreader
from collections import OrderedDict 

class CXRay(Cell):

    def find_closest_to_center(self,img, arr):
        arr = np.abs(arr - img.shape[1]/2)
        idx = np.argmin(arr)
        return idx    


    def extract_soft_tissue(self,img, lcl, rcl, mcl):

        diaphragm = img[img.shape[1]/2-1:img.shape[1] -1,max(0,lcl):rcl]

        pad = (img.shape[1] - diaphragm.shape[1])
        pad_n = pad/2
        if pad % 2 == 1:
            pad_n = pad/2 + 1
        #print rcl,pad
        dia_padded = cv2.copyMakeBorder(diaphragm, top=0, bottom = 0, left=pad_n, right=pad/2, borderType=cv2.BORDER_CONSTANT, value=0 )
        print dia_padded.shape
        heart = img[img.shape[0]/4:3*img.shape[0]/4, img.shape[0]/4:3*img.shape[0]/4]
        return (heart, dia_padded)    

    def mean_rows(self,img):
        sum_rows=np.zeros((1,512))
        for i in img:
            sum_rows+=i
        a=(sum_rows/512) + 128
        pks = argrelmax(a[0], order=50)
        return pks[0]


    def extract_lung_patches(self,img, peaks):
        sh = peaks.shape[0]
        if sh == 1:
            lb = 0
            c_l = peaks[0] + 10
            c_r = peaks[0] - 10
            rb = img.shape[1]
            #left = img[:,0:peaks[0]+10]
            #right = img[:,peaks[0]-10:]
        elif sh > 1:
            split = self.find_closest_to_center(img,peaks)
            if split == sh -1:
                lb = max(peaks[0]-10, 0)
                c_l=peaks[split]+10
                c_r = peaks[split]-10
                rb = img.shape[1]
                #left = img[:,max(peaks[0]-10, 0):peaks[split]+10]
                #right = img[:,peaks[split]-10:]
            elif split == 0:
                lb = 0
                c_l = peaks[split]+10
                c_r = peaks[split]-10
                rb = min(peaks[sh-1]+10, img.shape[1])
                #left = img[:,0:peaks[split]+10]
                #right = img[:,peaks[split]-10:min(peaks[sh -1]+10,img.shape[1])]
            else:
                lb = max(peaks[0]-10,0)
                c_l = peaks[split]+10
                c_r = peaks[split]-10
                rb = min(peaks[sh -1]+10,img.shape[1])
                #print peaks
                #left = img[:,max(peaks[0]-10, 0):peaks[split]+10] 
                #right = img[:,peaks[split]-10:min(peaks[sh -1]+10,img.shape[1])]
        else:
            lb = 0
            c_l = img.shape[1]/2
            c_r = img.shape[1]/2
            r_b = img.shape[1]-1
            
            #left = img[:,0:img.shape[1]/2]
            #right = img[:, img.shape[1]/2: img.shape[1]]
        left = img[:, lb:c_l]
        right = img[:, c_r:rb]
                  
        l_pad = (3*(img.shape[1]/4) - left.shape[1])
        r_pad = (3*(img.shape[1]/4) - right.shape[1])
        #print left.shape[1],l_pad, r_pad
        l_new = l_pad/2
        if l_pad%2 == 1:
            l_new = (l_pad/2) + 1
        r_new = r_pad/2
        if r_pad%2 == 1:
            r_new = (r_pad/2) + 1   

        left_pad = cv2.copyMakeBorder(left, top=0, bottom = 0, left=l_new, right=l_pad/2, borderType=cv2.BORDER_CONSTANT, value=0 )
        right_pad = cv2.copyMakeBorder(right, top=0, bottom = 0, left=r_pad/2, right=r_new, borderType=cv2.BORDER_CONSTANT, value=0 )  

        return (lb,c_l,c_r,rb,left_pad, right_pad)

    def split_image(self,img, name):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        peaks = self.mean_rows(gray)
        lb,c_l,c_r, rb,left, right = self.extract_lung_patches(gray, peaks)
        heart, dia = self.extract_soft_tissue(gray, lb, rb, (c_l+c_r)/2 )
        img_patches=OrderedDict()
        #print left.shape[1]
        img_patches[name.split('.')[0]+'_left_'+str(0)+'_'+str(lb)+'_'+str(c_l-lb)+'_'+str(left.shape[0])] = left
        img_patches[name.split('.')[0]+'_right_'+str(0)+'_'+str(c_r)+'_'+str(rb - c_r)+str(right.shape[0])] = right
        img_patches[name.split('.')[0]+'_heart_'+str(img.shape[0]/4)+'_'+str(3 * img.shape[0] / 4)+'_'+str(heart.shape[0])+'_'+str(heart.shape[0])] = heart
        img_patches[name.split('.')[0]+'_diaphragm_'+str(img.shape[0]/2) + '_' + str(lb)+'_'+str(rb-lb)+str(heart.shape[0])] = dia
        return img_patches

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
        	patches = self.img
            img_patches = self.split_image(self.curr_img, input_reader)	
		    out_patches.update(img_patches)
		    out_attribs["name"]=self.curr_img_name.split('.')[0]
		    return (out_patches, out_attribs)