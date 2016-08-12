''' cell.py: An abstract class for extracting patches of specific types'''

__author__="Rohit Kumar Pandey"
__copyright__="Copyright 2015, SigTuple Technologies Pvt Ltd"
__version__="0.1"
__maintainer__="Rohit Kumar Pandey"
__email__="rohit@sigtuple.com"
__status__="development"

import numpy as np
import cv2
import sys
import os
from scipy import ndimage
import math
from cell import Cell
from modules.inputreaders import baseinputreader

class Wbc(Cell):
    
    def __init__(self, config, label, attrib_flag, out_ppm, logger):
        self.config = config
        self.label = label
        self.logger = logger
        self.out_ppm = out_ppm
        self.attrib_flag = attrib_flag
        #load basic config params
        self.input_patch_size = self.config['patch_size']
        self.input_stride_size = self.config['stride_size']
        self.input_magn = self.config['magn']
        self.curr_img = None
        self.curr_img_name = None
        
    def pre_processing(self):
        
        self.logger.debug("Preprocessing the image with shape:"+str(self.curr_img.shape))

        #convert to HSV
        conv_img = cv2.cvtColor(self.curr_img,cv2.COLOR_BGR2HSV)
        
        bins = self.config["bins"]
        
        #use saturation plane only and threshold
        im_bw = cv2.threshold(conv_img[:,:,1],bins,255,cv2.THRESH_BINARY)[1]
        
        self.logger.debug("Preprocessing compelted for the image with shape:"+str(im_bw.shape))
        
        return im_bw
        
    def identify_objects(self, img):
        
        self.logger.debug("Invoked the function to identify objects for the image with shape:"+str(img.shape))
        
        kernel_size_x = self.config["kernel_size_x"]
        kernel_size_y = self.config["kernel_size_y"]
       
        #fill gaps 
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(kernel_size_x),int(kernel_size_y)))
        res = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
        
        self.logger.debug("Object identification completed for the image with shape:"+str(res.shape))
        
        return res
        
    def find_centroids(self, img, top_x, top_y):
       
        areas = {}
        diameters = {}
        rgbmeans = {}
        min_area = int(self.config["min_area_um"]*(self.out_ppm[0]*self.out_ppm[1]))
        max_area = int(self.config["max_area_um"]*(self.out_ppm[0]*self.out_ppm[1]))
        
        self.logger.debug("Finding centroids for image with shape:"+str(img.shape)+" and for area range:"+str(min_area)+","+str(max_area))
        
        #find contours and their centroids
        centroids = []
        ret,thresh = cv2.threshold(img,127,255,0)
        contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]

        #blue patch ratios
        sniffer_r_b_max_ratio = self.config['sniffer_r_b_max_ratio']
        sniffer_g_b_max_ratio = self.config['sniffer_g_b_max_ratio']
        sniffer_min_b_val = self.config.get('sniffer_min_b_val', 0)

        #create mask for contours
        contour_mask = np.zeros(img.shape, np.uint8)

        for cnt in contours:
            #check area
            area = cv2.contourArea(cnt)
            if area <= min_area or area >= max_area:
                continue
            #check if there's a blue patch in the contour
            contour_mask.fill(0)
            cv2.drawContours(contour_mask, [cnt], 0, 255, -1)
            (bm,gm,rm) = cv2.mean(self.curr_img, contour_mask)[:3]
            if bm <= sniffer_min_b_val or rm/bm > sniffer_r_b_max_ratio or gm/bm > sniffer_g_b_max_ratio:
                continue
            #valid contour
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            centroids.append([cy, cx])
            if self.attrib_flag:
                attrib_key = self.get_attribute_key((top_x+cx), (top_y+cy))
                areas[attrib_key] = area
                diameters[attrib_key] = 2.0*(np.sqrt(area/np.pi))/self.out_ppm[0]
                rgbmeans[attrib_key] = {'red': rm, 'blue': bm, 'green': gm}
        
        self.logger.debug("Total centroids extracted before duplicate check:"+str(len(centroids)))
        self.logger.debug("Centroids: %s" % ','.join(['(%d,%d)' % (top_x+x,top_y+y) for (y,x) in centroids]))
        
        dup_dict = {}
        dup_box_size_x = int(self.config["duplicate_box_size_um"]*self.out_ppm[0])
        dup_box_size_y = int(self.config["duplicate_box_size_um"]*self.out_ppm[1])
       
        #remove "duplicate" centroids - duplicate because if close enough,
        #they are expected to be 2 regions of same cell
        if dup_box_size_x > 0 and dup_box_size_y > 0:
            unique_centroids = []
            for (y,x) in centroids:
                x00,y00 = math.floor(x/dup_box_size_x), math.floor(y/dup_box_size_y)
                keys = [(x00,y00),(x00-0.5,y00),(x00+0.5,y00),\
                            (x00,y00-0.5),(x00,y00+0.5)]
                isadd = [False,False,False,False,False]
                ignore = False

                #check duplicates
                for i in xrange(len(keys)):
                    if dup_dict.get(keys[i], None):
                        if not ignore:
                            (xt,yt) = dup_dict[keys[i]]
                            self.logger.debug('ignoring centroid (%d,%d) due to centroid (%d,%d)' % (top_x+x,top_y+y,top_x+int(xt),top_y+int(yt)))
                        ignore = True
                    else:
                        isadd[i] = True

                if not ignore:
                    unique_centroids.append((y,x))
                    for i in xrange(len(isadd)):
                        if isadd[i]:
                            dup_dict[keys[i]] = (x,y)
                else:
                    if self.attrib_flag:
                        key = self.get_attribute_key((top_x+x),(top_y+y))
                        if key in diameters:
                            diameters.pop(key)
                        if key in rgbmeans:
                            rgbmeans.pop(key)
            centroids = unique_centroids
        
        self.logger.debug("Total centroids returned after duplicate check:"+str(len(centroids)))
        
        return (centroids,diameters,areas,rgbmeans)

    def get_patches_from_centroids(self,centroids,init_x,init_y,sreader):
        
        self.logger.debug("Extracting patches for the centroids")
        
        box_y = int(self.config["out_patch_height_um"]*self.out_ppm[1])
        box_x = int(self.config["out_patch_width_um"]*self.out_ppm[0])
        img_patches = {}
        
        height,width = self.curr_img.shape[:2]

        (in_sizex,in_sizey) = sreader.get_size()

        blue_tint_r_b_max_ratio = self.config['blue_tint_r_b_max_ratio']
        blue_tint_g_b_max_ratio = self.config['blue_tint_g_b_max_ratio']
        pink_tint_b_r_min_ratio = self.config['pink_tint_b_r_min_ratio']
        pink_tint_g_r_max_ratio = self.config['pink_tint_g_r_max_ratio']

        is_valid_stitch_type = True
        stitch_detection_types = set(self.config.get('stitch_detection_types', []))
        if sreader.get_type() not in stitch_detection_types:
            is_valid_stitch_type = False
   
        for (y,x) in centroids:
            level = 0
            centx = init_x + x
            centy = init_y + y
            #calculate patch location and size
            px = centx - (int(box_x)*2)
            py = centy - (int(box_y)*2)
            pxsize = 4*int(box_x)
            pysize = 4*int(box_y)
            img_s = sreader.get_patch(px, py, pxsize, pysize)
            is_border_patch = False
            if px < 0 or py < 0 or px+pxsize > in_sizex or py+pysize > in_sizey:
                is_border_patch = True
            
            rmean = np.mean(img_s[:,:,2])
            gmean = np.mean(img_s[:,:,1])
            bmean = np.mean(img_s[:,:,0])
          
            #check if blue tint in patch
            if bmean > 0 and rmean/bmean < blue_tint_r_b_max_ratio and gmean/bmean < blue_tint_g_b_max_ratio:
                self.logger.debug('not writing centroid (%d,%d): blue tint detected' % (centx,centy))
                continue
            #check if pink tint in patch
            if rmean > 0 and bmean/rmean > pink_tint_b_r_min_ratio and gmean/rmean < pink_tint_g_r_max_ratio:
                self.logger.debug('not writing centroid (%d,%d): pink tint detected' % (centx,centy))
                continue
            
             
            is_stitch = False    
            cell_cnt = 0
            x1 = (img_s.shape[0])/2 - int(box_x)
            x2 = (img_s.shape[0])/2 + int(box_x)
            y1 = (img_s.shape[0])/2 - int(box_y)
            y2 = (img_s.shape[0])/2 + int(box_y)
            wbc_patch = img_s[x1:x2,y1:y2]
                
            #stitch detection only for non-border patches, to 
            #avoid detection of image boundary as a stitch
            if is_valid_stitch_type and not is_border_patch:
                is_stitch = self.detect_stitch(wbc_patch)
            
            #remove patch if a stitch is detected (artifact of full scan of slide images usually)
            if is_stitch:
                self.logger.debug('not writing centroid (%d,%d): stitch detected'  % (centx,centy))
                continue

            #remove patch if blur is detected
            if self.detect_blur(wbc_patch):
                self.logger.debug('not writing centroid (%d,%d): blur detected'  % (centx,centy))
                continue

            cell_cnt = self.get_num_cells_in_patch(wbc_patch)

            #remove patch if its identified to have more that one cells in it 
            if cell_cnt > 1:
                self.logger.debug('not writing centroid (%d,%d): multi cells detected' % (centx,centy))
                continue
            
            img_patches[sreader.get_name().split(".")[0]+"_"+self.label+"_"+str(centx)+"_"+str(centy)] = wbc_patch
            
        self.logger.debug("Total count for one patches extracted:"+str(len(img_patches)))
            
        return img_patches
        
    def get_num_cells_in_patch(self, img):
        
        img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        bins = int(self.config["bins"])
        img_bw = cv2.threshold(img_hsv[:,:,1],bins,255,cv2.THRESH_BINARY)[1]
        kernel_size_x = int(self.config["kernel_size_x"])
        kernel_size_y = int(self.config["kernel_size_y"])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size_x,kernel_size_y))
        img_fill = cv2.morphologyEx(img_bw,cv2.MORPH_OPEN,kernel)
        min_area = int(self.config["min_area_um"]*(self.out_ppm[0]*self.out_ppm[1]))
        max_area = int(self.config["max_area_um"]*(self.out_ppm[0]*self.out_ppm[1]))
        contours = cv2.findContours(cv2.threshold(img_fill,127,255,cv2.THRESH_BINARY)[1],cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2]
        
        cells = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area and area <= max_area:
                cells += 1
        return cells


    def detect_stitch(self, img):
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        thresh_low = int(self.config["canny_thresh_low"])
        thresh_high = int(self.config["canny_thresh_high"])
        aperture_size = int(self.config["canny_aperture"])

        edges = cv2.Canny(img_gray, thresh_low, thresh_high, apertureSize=aperture_size)

        hough_thresh = int(self.config["hough_threshold"])
        hough_rho = int(self.config["hough_rho"])
        hough_theta = int(self.config["hough_theta_degrees"])
        min_len_ratio = float(self.config["hough_min_len_ratio"])
        max_gap_ratio = float(self.config["hough_max_gap_ratio"])

        lines = cv2.HoughLinesP(edges, hough_rho, np.radians(hough_theta), hough_thresh, minLineLength=img.shape[0]*min_len_ratio, maxLineGap=img.shape[0]*max_gap_ratio)

        if None is lines:
            return False
        cnt = 0
        for l in lines:
            x1,y1,x2,y2  = l[0]
            if abs(x1-x2) < 2 or abs(y1-y2) < 2:
                self.logger.debug('stitch detected between (%d,%d) and (%d,%d)' % (x1,y1,x2,y2))
                cnt += 1
        return (cnt > 0)

    def detect_blur(self, img):
        blur_threshold = float(self.config.get('blur_threshold', 0.0))
        blur_val = cv2.Laplacian(img, cv2.CV_16S).var()
        if blur_val <= blur_threshold:
            return True
        return False

    def extract_patches(self, input_info, num_patches, all_global_attrib=False):
        if all_global_attrib and num_patches <= 0:
            return ({}, {})
        #initialize input reader
        input_reader = baseinputreader.factory(input_info['type'], self.logger, input_info['path'], in_ppm=[input_info['ppm_x'], input_info['ppm_y']], out_ppm=self.out_ppm)
        input_reader.reset(jump=self.input_stride_size, patch_size=self.input_patch_size, magn=self.input_magn)

        out_patches = {}
        out_attribs = {'diameter': {}, 'rgb': {}, 'area': {}}
        for (x,y,img) in input_reader:
            if len(out_patches) >= num_patches:
                break
            self.curr_img = img
            self.curr_img_name = input_reader.get_name()
            preproc_img = self.pre_processing()
            thresh_img = self.identify_objects(preproc_img)
            (centroids, diameters, areas, rgbmeans) = self.find_centroids(thresh_img, x, y)
            if self.attrib_flag:
                out_attribs['diameter'].update(diameters)
                out_attribs['rgb'].update(rgbmeans)
                out_attribs['area'].update(areas)
            patches = self.get_patches_from_centroids(centroids, x, y, input_reader)
            if patches and len(patches) > 0:
                out_patches.update(patches)
                self.logger.info('extracted %d %s from %s, required %d' % (len(out_patches), self.label, os.path.basename(input_info['path']), num_patches))
        input_reader.close()
        return (out_patches, out_attribs)
