#!/usr/bin/env python
''' rbc.py: to read the blood smear slides and extract rbc patches'''

import cv2
import numpy as np
import sys
import os
from scipy import ndimage
from scipy.spatial import distance
from cell import Cell
from modules.inputreaders import baseinputreader

__author__="Rohit Kumar Pandey"
__copyright__="Copyright 2015, SigTuple Technologies Pvt Ltd"
__version__="0.1"
__maintainer__="Rohit Kumar Pandey"
__email__="rohit@sigtuple.com"
__status__="development"


class Rbc(Cell):

    def __init__(self, config, label, attrib_flag, out_ppm, logger):
        self.config = config
        self.logger = logger
        self.label = label
        self.attrib_flag = attrib_flag
        self.out_ppm = out_ppm
        self.curr_img = None
        self.curr_img_name = None
        self.input_patch_size = self.config['patch_size']
        self.input_stride_size = self.config['stride_size']
        self.input_magn = self.config['magn']
        self.img_mask = None

    def pre_processing(self):

        self.logger.debug("Pre-processing of the image for rbc extraction started : Rbc. pre_processign()")

        n = int(self.config["mask"])

        if n%2 == 0:
            return (None,None,None)

        r = (n-1)/2
        y,x = np.ogrid[-r:n-r, -r:n-r]
        mask = x*x + y*y <= r*r
        array = np.zeros((n,n), dtype=np.uint8)
        array[mask] = 1
        gchannel = self.curr_img[:,:,1]
        blur = cv2.medianBlur(gchannel, 3)

        hist, bins = np.histogram(blur, 64, [0,256])
        area = self.curr_img.shape[0]*self.curr_img.shape[1]*0.005

        i = 63
        while (i > 0 and (hist[i] <= hist[i-1] or hist[i-1] < area)):
            i -= 1
        if i == 0:
            return (None, None, None)
        n2 = i

        while (i > 0 and (hist[i] >= hist[i-1] or n2-i < 10)):
            i -= 1
        if i == 0:
            return (None, None, None)
        central_min = i

        while (i >= 0 and hist[i] <= hist[i-1]):
            i -= 1
        if i == 0:
            return (None, None, None)

        n1 = i
        cmax = hist[i]

        while i > 10:
            if hist[i] > cmax:
                cmax = hist[i]
                n1 = i
            i -= 1
        if n1 == 0:
            return (None, None, None)

        c_min = -1
        u_t = None
        for u in range(n1+1, n2):
            c = 0
            for i in range(n1, u):
                c += (i-n1)*(i-n1)*hist[i]
            for i in range(u+1,n2):
                c += (i-n2)*(i-n2)*hist[i]
            if c_min < 0 or c < c_min:
                c_min = c
                u_t = u
        thresh, t_img = cv2.threshold(blur, u_t*4, 255, cv2.THRESH_BINARY_INV)

        self.logger.debug("Shape of the image returned by the pre-processing function:"+str(t_img.shape))
        return t_img,thresh,array

    def identify_objects(self, img, mask):

        self.logger.debug("Identification of objects started by invoking Rbc.identify_objects() for image with size:"+str(img.shape))

        inv = cv2.bitwise_not(img)
        r, l, s, c = cv2.connectedComponentsWithStats(inv, connectivity=8)

        sizes = []
        for i in range(len(c)):
            sizes.append(s[i][cv2.CC_STAT_AREA])

        sizes = np.sort(sizes)

        #default percentile is 98, however, if the size_percentile_threshold is
        #passed then it will consume it.
        #percentile_threshold = 0.98 if ('size_percentile_threshold' not in self.config or None is self.config['size_percentile_threshold']) else self.config['size_percentile_threshold']
        percentile_threshold = self.config.get('size_percentile_threshold', 0.98)
        thresh = sizes[len(sizes)*percentile_threshold]

        centers = np.zeros(img.shape, dtype=np.uint8)
        centers[l>0] = 255
        num_centers = 0

        for i in range(len(c)):
            size = s[i][cv2.CC_STAT_AREA]
            if size < thresh:
                num_centers += 1
                continue
            centers[l==i] = 0

        centers = cv2.dilate(centers, mask, iterations=1)

        if (centers == 255).all():
            return None,None

        hole_filled = cv2.add(img, centers)

        self.logger.debug("Shape of the image and centers returned by the Rbc.identify_objects fucntion:"+str(hole_filled.shape)+","+str(len(centers)))
        return hole_filled,centers


    def find_centroids(self, img, mask,centers,thresh_p, top_x, top_y):
        self.logger.debug("Finding the centroids for the image with image shape:"+str(len(img.shape)))
        
        cells_with_center,dprev = self.find_cells_with_center(img, centers,mask, top_x, top_y, thresh_p)
        clumps = self.find_cells_without_center(img, dprev, top_x, top_y)
        
        self.logger.debug("Total centroids returned for the image patch:"+str(len(self.centroids)))

        return self.centroids


    def get_patches_from_centroids(self,centroids,init_x,init_y,sreader):
        self.logger.debug("Extraction patches for the centroids")
        img_patches = {}
        box_y = int(self.config["out_patch_height_um"]*self.out_ppm[1])
        box_x = int(self.config["out_patch_width_um"]*self.out_ppm[0])
        r_box_x = box_x * 2
        r_box_y = box_y * 2

        blue_tint_r_b_max_ratio = self.config['blue_tint_r_b_max_ratio']
        blue_tint_g_b_max_ratio = self.config['blue_tint_g_b_max_ratio']
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
            rbc_patch = sreader.get_patch(centx-box_x, centy-box_y, r_box_x, r_box_y)
                   
            rmean = np.mean(rbc_patch[:,:,2])
            gmean = np.mean(rbc_patch[:,:,1])
            bmean = np.mean(rbc_patch[:,:,0])

            #check if blue tint in patch
            #if bmean > 0 and rmean/bmean < blue_tint_r_b_max_ratio and gmean/bmean < blue_tint_g_b_max_ratio:
            #    self.logger.debug('not writing centroid (%d,%d): blue tint detected' % (centx,centy))
            #    continue

            #check if pink tint in patch
            if rmean > 0 and bmean/rmean > pink_tint_b_r_min_ratio and gmean/rmean < pink_tint_g_r_max_ratio:
                self.logger.debug('not writing centroid (%d,%d): pink tint detected' % (centx,centy))
                continue

            if is_valid_stitch_type:
                is_stitch = self.detect_stitch(rbc_patch)
            if is_stitch:
                self.logger.debug("not writing the centroid :- (%d,%d):stitch detected" %(centx,centy))
                continue

            if self.config.get('remove_background', False):
                if not self.remove_background(rbc_patch, self.stats, self.pixel_labels, centx, centy):
                    self.logger.info('could not remove background for patch (%d,%d), ignoring' % (centx,centy))
 
            if int(rbc_patch.shape[0]) ==  int(r_box_x) and int(rbc_patch.shape[1]) == int(r_box_y):
                img_patches[sreader.get_name().split(".")[0]+"_rbc_"+str(centx)+"_"+str(centy)] = rbc_patch

        self.logger.debug("Total number of image patches extracted:"+ str(len(img_patches)))
        return img_patches


    def find_cells_with_center(self,holes_filled, centers, mask, top_x, top_y, thresh_p):
        done = False
        size_f = int(self.out_ppm[0]*self.out_ppm[1])

        dprev = centers

        while not done:
            d0 = cv2.dilate(dprev, mask, iterations=1)
            d1 = cv2.bitwise_and(d0, holes_filled)
            if (d1 == dprev).all():
                done = True
            dprev = d1

        r, l, s, c = cv2.connectedComponentsWithStats(dprev, connectivity=4)

        #sizes = []
        #for i in range(len(c)):
        #    sizes.append(s[i][cv2.CC_STAT_AREA])

        #sizes = np.sort(sizes)
        #computing max and min thresholds.
        #logic is to sort the areas of all connected components and reject
        #those which are not in the (max_area_thresh_pcnt,min_area_thresh_pcnt) range
        #
        #However, since the CC_STAT_AREA is in sq-pixels and all of our comaprison
        #should be in sq-microns. We need to convert the sq-pixels to sq-microns
        #this is done by multiplying the pixel values with the size_f (which denotes sq-pixels per sq-micron)
        #
        # That said. Below max_thresh and min_thresh denote the max_area and in min_area in sq-microns
        #
        #thresh = sizes[len(sizes)*self.config["max_area_thresh_pcnt"]] * size_f
        #min_thresh = sizes[len(sizes)*self.config["min_area_thresh_pcnt"]] * size_f
        # not using  the above as the percentile strategy is somehow, failing.
        # fixing it by accepting the min and max area via config

        cells_with_center = np.zeros(holes_filled.shape, dtype=np.uint8)

        count_cent = 0
        for i in range(len(c)):
            size = s[i][cv2.CC_STAT_AREA]

            #converting the area to sq-micros so that the comaprison is standardised
            size_in_um = self.sqpixel_to_sqmicron_converter(size)
            self.logger.debug("Area conversion from %d sq-pixel to %f sq-microns" % (size,size_in_um))

            #comapring if the size is within min and max percentile of all component sizes
            #if size > thresh or size < min_thresh:
            #    self.logger.debug("[CellsWithCenter] Ignoring the patch at %s as the size is %f " %(str(c[i]),size))
            #    continue

            #comparing the size in microns, with threshold of rbc area. (given that radii is between 2.5um to 5um the area could
            #be between 30 and 100. Made this a config)
            if size_in_um < self.config['min_cell_area_um'] or size_in_um > self.config['max_cell_area_um']:
                self.logger.debug("[CellsWithCenter] Ignoring the patch at %s as the size is %f " %(str(c[i]),size_in_um))
                continue

            if s[i][cv2.CC_STAT_TOP] == 0 or s[i][cv2.CC_STAT_LEFT] == 0:
                continue

            x = s[i][cv2.CC_STAT_LEFT]
            y = s[i][cv2.CC_STAT_TOP]
            h = s[i][cv2.CC_STAT_HEIGHT]
            w = s[i][cv2.CC_STAT_WIDTH]
            cell = dprev[y:y+h,x:x+w]
            label = np.array(l[y:y+h,x:x+w])
            label[label != i] = 0
            label[label == i] = 255
            label = np.array(label, dtype=np.uint8)
            hist_arr = []
            cell = cv2.bitwise_and(cell, label)
            image,contours,hierarchy = cv2.findContours(cell,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if None is contours:
                continue
            #choose max area contour instead of taking the first one arbitrarily
            cnt = None
            max_cntarea = 0
            for tempcnt in contours:
                cntarea = cv2.contourArea(tempcnt)
                if cntarea > max_cntarea:
                    max_cntarea = cntarea
                    cnt = tempcnt
            if None is not cnt and max_cntarea > 0:
                cent_x,cent_y = int(c[i][0]),int(c[i][1])
                if self.attrib_flag:
                    i_mask = np.zeros(cell.shape,np.uint8)
                    cv2.drawContours(i_mask,[cnt],0,255,-1)
                    #s_img = cv2.bitwise_and(self.curr_img,self.curr_img,mask=i_mask)
                    #do not consider pallor area for rgb mean
                    #get orig image cell area
                    cell_image = self.curr_img[y:y+h,x:x+w]
                    green_plane_cell = cv2.bitwise_and(cell_image[:,:,1], cell_image[:,:,1], mask=i_mask)
                    pallor_region_mask = cv2.threshold(green_plane_cell, thresh_p-15, 255, cv2.THRESH_BINARY)[1]
                    #self.global_attribs['pallorarea'] = self.global_attribs.get('pallorarea', 0.0)+pallor_region_mask[pallor_region_mask!=0].size
                    #subtract pallor region from mask
                    i_mask -= pallor_region_mask
                    (bm,gm,rm) = cv2.mean(cell_image, i_mask)[:3]
                    attrib_key = self.get_attribute_key((top_x+cent_x), (top_y+cent_y))
                    self.area[attrib_key] = int(size)
                    self.diameters[attrib_key] = self.get_diameter_from_contour(cnt, int(size))
                    self.stats[attrib_key] = [x,y,h,w,i,0]
                    self.rgbmeans[attrib_key] = {'red': rm, 'green': gm, 'blue': bm}
                    self.ratio[attrib_key] = (pallor_region_mask.sum()/255.)/size
                count_cent += 1
                self.centroids.append([cent_y,cent_x])
            cells_with_center[l==i] = 255

        self.pixel_labels[0] = l
        self.logger.debug("Total centroids extract from cell with centers:"+str(count_cent))
        return (cells_with_center,dprev)

    #utility that converts sq-pixel area to sq-micron area
    def sqpixel_to_sqmicron_converter(self,pix_val):
        self.logger.debug("Converting pixel to ppm. Details : (pix_val=%d), (out_ppm=%s)" % (pix_val,str(self.out_ppm)))
        if not self.out_ppm or not self.out_ppm[0] or not self.out_ppm[1]:
            raise ValueError("Invalid ppm value. Details : (pix_val=%d), (out_ppm=%s)" % (pix_val,str(self.out_ppm)))
        x_ppm = self.out_ppm[0]
        y_ppm = self.out_ppm[1]
        return float(pix_val/(x_ppm * y_ppm))

    def find_cells_without_center(self,hole_filled, dprev, top_x, top_y):
        self.logger.debug("Finding centroids for cells without centers. Invoked Rbc.find_cells_without_center()")
        cells_without_holes = cv2.subtract(hole_filled,dprev)
        clumps = np.zeros(hole_filled.shape, dtype=np.uint8)

        r, l, s, c = cv2.connectedComponentsWithStats(cells_without_holes, connectivity=8)
        cells_with_center = np.zeros(hole_filled.shape, dtype=np.uint8)

        count_cent = 0
        for i in range(len(c)):
            size = s[i][cv2.CC_STAT_AREA]
            size_in_um = self.sqpixel_to_sqmicron_converter(size)

            self.logger.debug("Area conversion from %d sq-pixel to %f sq-microns" % (size,size_in_um))

            #comapring if the size is within min and max percentile of all component sizes
            #if size > thresh or size < min_thresh:
            #    self.logger.debug("[CellWithoutCenter] Ignoring the patch at %s as the size is %f " %(str(c[i]),size))
            #    continue

            #comparing the size in microns, with threshold of rbc area. (given that radii is between 2.5um to 5um the area could
            #be between 30 and 100. Made this a config)
            if size_in_um < self.config['min_cell_area_um'] or size_in_um > self.config['max_cell_area_um']:
                self.logger.debug("[CellWithoutCenter] Ignoring the patch at %s as the size is %f " %(str(c[i]),size_in_um))
                continue

            if s[i][cv2.CC_STAT_TOP] == 0 or s[i][cv2.CC_STAT_LEFT] == 0:
                continue

            x = s[i][cv2.CC_STAT_LEFT]
            y = s[i][cv2.CC_STAT_TOP]
            h = s[i][cv2.CC_STAT_HEIGHT]
            w = s[i][cv2.CC_STAT_WIDTH]

            cell = cells_without_holes[y:y+h,x:x+w]
            image,contours,hierarchy = cv2.findContours(cell,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if None is contours:
                continue
            cnt = None
            max_cntarea = 0
            #find max area contour instead of picking first one arbitrarily
            for tempcnt in contours:
                cntarea = cv2.contourArea(tempcnt)
                if cntarea > max_cntarea:
                    max_cntarea = cntarea
                    cnt = tempcnt
            #cnt = contours[0]

            if None is not cnt and max_cntarea > 0:
                cent_x,cent_y=int(c[i][0]),int(c[i][1])
                if self.attrib_flag:
                    attrib_key = self.get_attribute_key((top_x+cent_x), (top_y+cent_y))
                    i_mask = np.zeros(cell.shape,np.uint8)
                    cv2.drawContours(i_mask,[cnt],0,255,-1)
                    (bm,gm,rm) = cv2.mean(self.curr_img[y:y+h,x:x+w],i_mask)[:3]
                    #s_img = cv2.bitwise_and(self.curr_img,self.curr_img,mask=i_mask)
                    self.diameters[attrib_key] = self.get_diameter_from_contour(cnt, int(size))
                    self.area[attrib_key]=int(size)
                    self.stats[attrib_key]=[x,y,h,w,i,1]
                    self.rgbmeans[attrib_key]={'red': rm, 'blue': bm, 'green': gm}
                    self.ratio[attrib_key] = 0.0
                count_cent += 1
                self.centroids.append([cent_y,cent_x])
            clumps[l==i] = 255
        self.logger.debug("Total centroids extract from cell without centers:"+str(count_cent))
        self.pixel_labels[1] = l
        return clumps

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
                cnt += 1
        return (cnt > 0)

    def extract_patches(self, input_info, num_patches, all_global_attrib=False):
        #initialize input reader
        input_reader = baseinputreader.factory(input_info['type'], self.logger, input_info['path'], in_ppm=[input_info['ppm_x'], input_info['ppm_y']], out_ppm=self.out_ppm)
        input_reader.reset(jump=self.input_stride_size, patch_size=self.input_patch_size, magn=self.input_magn)

        out_patches = {}
        out_attribs = {'diameter': {}, 'area': {}, 'ratio': {}, 'rgb': {}}
        self.global_attribs = {'bgdarea': 0.0, 'rbcarea': 0.0}
        for (x,y,img) in input_reader:
            if len(out_patches) >= num_patches and not all_global_attrib:
                break
            self.curr_img = img
            self.curr_img_name = input_reader.get_name()
            self.area = {}
            self.diameters = {}
            self.rgbmeans = {}
            self.stats = {}
            self.ratio = {}
            self.centroids = []
            self.pixel_labels = [None,None]
            conv_img, thresh, mask = self.pre_processing()
            if None is conv_img or not conv_img.any():
                continue
            thresh_img, centers = self.identify_objects(conv_img, mask)
            if self.attrib_flag:
                self.global_attribs['bgdarea'] = self.global_attribs.get('bgdarea', 0.0) + thresh_img[thresh_img==0].size
                background_mask = np.zeros(img.shape, dtype=np.uint8)
                #ignore black surrounding of microscope fov
                background_mask[img[:,:,1] > 40] = 255
                background_mask = cv2.bitwise_and(background_mask, background_mask, mask=thresh_img)
                self.global_attribs['rbcarea'] = self.global_attribs.get('rbcarea', 0.0) + background_mask[background_mask==255].size
            if len(out_patches) < num_patches:
                centroids = self.find_centroids(thresh_img, mask, centers, thresh, x, y)
                patches = self.get_patches_from_centroids(centroids, x, y, input_reader)
                if patches and len(patches) > 0:
                    out_patches.update(patches)
                    out_attribs['diameter'].update(self.diameters)
                    out_attribs['area'].update(self.area)
                    out_attribs['ratio'].update(self.ratio)
                    out_attribs['rgb'].update(self.rgbmeans)
                    self.logger.info('extracted %d %s from %s, required %d' % (len(out_patches), self.label, os.path.basename(input_info['path']), num_patches))
        input_reader.close()
        #update region areas in attribs
        if self.attrib_flag:
            out_attribs.setdefault('global',{})[os.path.splitext(input_reader.get_name())[0]] = {'bgd_area': self.global_attribs.get('bgdarea', 0.0), 
                                                'rbc_area': self.global_attribs.get('rbcarea', 0.0)}
        return (out_patches, out_attribs)

    def remove_background(self, patch, stats_info, pixel_labels, x, y):
        if None is stats_info or None is pixel_labels:
            return False
        box_y = patch.shape[0]/2#int(self.config["out_patch_height_um"]*self.out_ppm[1])
        box_x = patch.shape[1]/2#int(self.config["out_patch_width_um"]*self.out_ppm[0])
        r_box_x = box_x * 2
        r_box_y = box_y * 2

        if None is self.img_mask:
            self.img_mask = np.zeros((r_box_y, r_box_x), dtype=np.uint32)

        img_wd = pixel_labels[0].shape[1]
        img_ht = pixel_labels[0].shape[0]

        #get stats
        stats = stats_info.get('%d_%d' % (x,y), None)
        if None is stats:
            return False
        self.img_mask.fill(stats[4]+1)
        #fill mask from labels
        mx, my = (x%self.input_stride_size)-box_x, (y%self.input_stride_size)-box_y
        mx1,mx2 = max(0,mx), min(mx+r_box_x, img_wd)
        my1,my2 = max(0,my), min(my+r_box_y, img_ht)
        self.img_mask[max(0,0-my):max(0,0-my)+(my2-my1),max(0,0-mx):max(0,0-mx)+(mx2-mx1)] = pixel_labels[stats[5]][my1:my2,mx1:mx2]
        patch[self.img_mask!=stats[4]] = 0
        return True

    def get_background_removal_info(self, input_reader, x, y):
        #get image patch to process
        x0 = x-x%self.input_stride_size
        y0 = y-y%self.input_stride_size
        self.curr_img = input_reader.get_patch(x0, y0, self.input_patch_size, self.input_patch_size)
        self.area = {}
        self.rgbmeans = {}
        self.stats = {}
        self.ratio = {}
        self.centroids = []
        self.pixel_labels = [None,None]
        conv_img, thresh, mask = self.pre_processing()
        if None is conv_img or not conv_img.any():
            return (None, None)
        thresh_img, centers = self.identify_objects(conv_img, mask)
        centroids = self.find_centroids(thresh_img, mask, centers, thresh, x0, y0)
        return (self.stats, self.pixel_labels)

    def get_diameter_from_contour(self, cntr, area):
        #approximate cell to be a circle and use area for diameter
        return (2*np.sqrt(area/np.pi))/self.out_ppm[0]
        
        #min_rect = cv2.minAreaRect(cntr)
        #box = np.int0(cv2.boxPoints(min_rect))
        #side_lengths = []
        #for i in xrange(1, len(box)):
        #    side_lengths.append(distance.euclidean(box[0], box[1]))
        #side_lengths.sort(reverse=True)
        #return side_lengths[1]/self.out_ppm[0];
