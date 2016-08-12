#!/usr/bin/env python
''' malaria.py: to read the blood smear slides and extract malaria infected RBC patches'''

import cv2
import numpy as np
import sys
import os
from scipy import ndimage
from rbc import Rbc
from modules.inputreaders import baseinputreader

__author__="Bharath Cheluvaraju"
__copyright__="Copyright 2015, SigTuple Technologies Pvt Ltd"
__version__="0.1"
__email__="bharath@sigtuple.com"


class Malaria(Rbc):

    def __init__(self, config, label, attrib_flag, out_ppm, logger):
        self.config = config
        self.logger = logger
        self.label = label
        self.attrib_flag = attrib_flag
        self.out_ppm = out_ppm
        self.curr_img = None
        self.input_patch_size = self.config['patch_size']
        self.input_stride_size = self.config['stride_size']
        self.input_magn = self.config['magn']

    def get_pixels_of_interest(self,img,(min_x,max_x),(min_y,max_y),blue_tint_r_b_range,blue_tint_g_b_range):
        count = 0
        for i in range(min_y,max_y):
            for j in range(min_x,max_x):
                (b,g,r) = img[i,j,:]
                if b < r or r < g or b < int(255*blue_tint_g_b_range[0]):
                    continue
                r_to_b = r*1./b
                g_to_b = g*1./b
                if r_to_b < blue_tint_r_b_range[0] or r_to_b > blue_tint_r_b_range[1]:
                    continue
                if g_to_b < blue_tint_g_b_range[0] or g_to_b > blue_tint_g_b_range[1]:
                    continue
                count = count +1
        return count



    def get_patches_from_centroids(self,centroids,init_x,init_y,sreader):
        self.logger.debug("Extraction patches for the centroids")
        img_patches = {}
        box_y = int(self.config["out_patch_height_um"]*self.out_ppm[1])
        box_x = int(self.config["out_patch_width_um"]*self.out_ppm[0])
        r_box_x = box_x * 2
        r_box_y = box_y * 2

        blue_tint_r_b_range = self.config['blue_tint_r_b_range']
        blue_tint_g_b_range = self.config['blue_tint_g_b_range']

        pink_tint_b_r_min_ratio = self.config['pink_tint_b_r_min_ratio']
        pink_tint_g_r_max_ratio = self.config['pink_tint_g_r_max_ratio']


        is_valid_stitch_type = True
        stitch_detection_types = set(self.config.get('stitch_detection_types', []))
        if sreader.get_type() not in stitch_detection_types:
            is_valid_stitch_type = False

        for (y,x) in centroids:
            is_stitch = False
            level = 0
            centx = init_x + x
            centy = init_y + y
            img_s = sreader.get_patch(centx-box_x, centy-box_y, r_box_x, r_box_y)

            x1 = (img_s.shape[0])/2 - int(box_x)
            x2 = (img_s.shape[0])/2 + int(box_x)
            y1 = (img_s.shape[0])/2 - int(box_y)
            y2 = (img_s.shape[0])/2 + int(box_y)
            plausible_malaria = img_s[x1:x2,y1:y2]
            count = self.get_pixels_of_interest(img_s,(x1,x2),(y1,y2), blue_tint_r_b_range,blue_tint_g_b_range)
            if count <= 15:
                continue

            img_tmp = img_s[img_s.shape[0]/2, img_s.shape[0]/2]
            bv,gv,rv = img_tmp[0],img_tmp[1],img_tmp[2]

            rmean = np.mean(img_s[:,:,2])
            gmean = np.mean(img_s[:,:,1])
            bmean = np.mean(img_s[:,:,0])

            #check if pink tint in patch
            if rmean > 0 and bmean/rmean > pink_tint_b_r_min_ratio and gmean/rmean < pink_tint_g_r_max_ratio:
                self.logger.debug('not writing centroid (%d,%d): pink tint detected' % (centx,centy))
                continue

            malaria_patch = img_s
            if is_valid_stitch_type:
                is_stitch = self.detect_stitch(malaria_patch)
            if is_stitch:
                self.logger.debug("not writing the centroid :- (%d,%d):stitch detected" %(centx,centy))
                continue
            if int(malaria_patch.shape[0]) ==  int(r_box_x) and int(malaria_patch.shape[1]) == int(r_box_y):
                img_patches[sreader.get_name().split(".")[0]+"_malaria_"+str(centx)+"_"+str(centy)] = malaria_patch

        self.logger.debug("Total number of image patches extracted:"+ str(len(img_patches)))
        return img_patches


    def extract_patches(self, input_info, num_patches, all_global_attrib=False):
        if all_global_attrib and num_patches <= 0:
            return ({}, {})
        #initialize input reader
        input_reader = baseinputreader.factory(input_info['type'], self.logger, input_info['path'], in_ppm=[input_info['ppm_x'], input_info['ppm_y']], out_ppm=self.out_ppm)
        input_reader.reset(jump=self.input_stride_size, patch_size=self.input_patch_size, magn=self.input_magn)

        out_patches = {}
        out_attribs = {'area': {}, 'ratio': {}}
        for (x,y,img) in input_reader:
            if len(out_patches) >= num_patches:
                break
            self.curr_img = img
            self.area = {}
            self.stats = {}
            self.ratio = {}
            self.centroids = []
            conv_img, thresh, mask = self.pre_processing()
            if None is conv_img or not conv_img.any():
                continue
            thresh_img, centers = self.identify_objects(conv_img, mask)
            centroids = self.find_centroids(thresh_img, mask, centers, thresh, x, y)
            patches = self.get_patches_from_centroids(centroids, x, y, input_reader)
            if patches and len(patches) > 0:
                out_patches.update(patches)
                if self.attrib_flag:
                    out_attribs['area'].update(self.area)
                    out_attribs['ratio'].update(self.ratio)
                self.logger.info('extracted %d %s from %s, required %d' % (len(out_patches), self.label, os.path.basename(input_info['path']), num_patches))
        input_reader.close()
        return (out_patches, out_attribs)
