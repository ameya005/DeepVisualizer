import os
import openslide
import baseinputreader
import random
import numpy
import cv2
from PIL import Image
from modules.utils.fsutils import DataFile
import math

class SlideReader(baseinputreader.BaseInputReader):
    def __init__(self, logger, inpath, jump=1024, magn=0, patch_size=1024, in_ppm=(1.0,1.0), out_ppm=(1.0,1.0)):
        self.logger = logger
        self.inpath = inpath
        try:
            #fetch from S3 if reqd
            infp = DataFile(inpath, 'r', self.logger).get_fp()
            self.slide = openslide.OpenSlide(inpath)
        except Exception as e:
            self.logger.error('could not open slide at %s, error: %s' % (inpath, str(e)))
            raise e
        finally:
            infp.close()

        if not self.slide:
            raise Exception('could not open slide at %s' % inpath)
        self.ind_x = 0
        self.ind_y = 0
        if magn < 0 or magn >= len(self.slide.level_dimensions):
            raise Exception('unsupport magnification %d for slide %s' % (magn, inpath))
        self.magn = magn
        self.in_ppm = in_ppm
        self.out_ppm = out_ppm
        self.scale_x = self.out_ppm[0]/self.in_ppm[0]
        self.scale_y = self.out_ppm[1]/self.in_ppm[1]
        self.patch_size_x = int(math.ceil(patch_size/self.scale_x))
        self.patch_size_y = int(math.ceil(patch_size/self.scale_y))
        self.jump_x = int(math.ceil(jump/self.scale_x))
        self.jump_y = int(math.ceil(jump/self.scale_y))

        self.size_x,self.size_y = self.slide.level_dimensions[self.magn]

    def get_type(self):
        return 'slide'

    def get_name(self):
        return os.path.basename(self.inpath)

    def get_size(self):
        return (self.size_x, self.size_y)

    def get_properties(self):
        props = {}

        props["res_x"], props["res_y"] = self.slide.level_dimensions[0]
        props["magn"] = self.slide.properties["tiff.ImageDescription"].split(" ")[1].split("=")[1]
        if 'openslide.mpp-x' in self.slide.properties:
            props["ppm_x"] = 1.0/float(self.slide.properties["openslide.mpp-x"])
        if 'openslide.mpp-y' in self.slide.properties:
            props["ppm_y"] = 1.0/float(self.slide.properties["openslide.mpp-y"])

        return props

    def next(self):
        #TODO: solve this cleanly
        #heuristic to disregard area of slide with clumped RBCs 
        #this area is not considered for analysis by pathologists
        #so we should also do the same
        clean_size_x = self.size_x
        if self.size_x > 40000:
            clean_size_x = 30000
        #get next patch
        if self.ind_y >= self.size_y:
            self.ind_y = 0
            self.ind_x += self.jump_x
        if self.ind_x >= clean_size_x:
            raise StopIteration
        pimg = self.slide.read_region((self.ind_x,self.ind_y), self.magn, (self.patch_size_x, self.patch_size_y))
        out_x, out_y = self.ind_x, self.ind_y
        self.ind_y += self.jump_y
        pimg = pimg.convert('RGB')
        pimg = numpy.asarray(pimg)
        #scale
        pimg = cv2.resize(pimg[:,:,[2,1,0]], (0, 0), fx=self.scale_x, fy=self.scale_y)
        #return cv2 default format BGR
        return (out_x, out_y, pimg)

    def reset(self, jump=-1, patch_size=-1, magn=-1, out_ppm=None):
        self.ind_x = 0
        self.ind_y = 0
        if jump > 0:
            self.logger.info('changing jump value to %d' % (jump))
            self.jump_x = int(math.ceil(jump/self.scale_x))
            self.jump_y = int(math.ceil(jump/self.scale_y))
        if patch_size > 0:
            self.logger.info('changing patch_size to %d' % (patch_size))
            self.patch_size_x = int(math.ceil(patch_size/self.scale_x))
            self.patch_size_y = int(math.ceil(patch_size/self.scale_y))
        if magn >= 0 and magn < len(self.slide.level_dimensions) and magn != self.magn:
            self.logger.info('changing magnification value from %d to %d' % (self.magn, magn))
            self.magn = magn
            self.size_x, self.size_y = self.slide.level_dimensions[self.magn]
        if out_ppm:
            self.out_ppm = out_ppm
            self.scale_x = self.out_ppm[0]/self.in_ppm[0]
            self.scale_y = self.out_ppm[1]/self.in_ppm[1]

    def get_patch(self, x, y, size_x, size_y):
        if x > self.size_x or y > self.size_y:
            raise Exception('provided coordinates (%d,%d) lie outside the image size (%d,%d)' % (x,y,self.size_x,self.size_y))
        nsize_x = int(math.ceil(size_x/self.scale_x))
        nsize_y = int(math.ceil(size_y/self.scale_y))
        pimg = self.slide.read_region((x,y), self.magn, (nsize_x, nsize_y))
        pimg = pimg.convert('RGB')
        pimg = numpy.asarray(pimg)
        return cv2.resize(pimg[:,:,[2,1,0]], (size_y, size_x))

    def get_cumulative_histogram(self):
        total_hist = None
        total_cnt = 0.0

        for i in xrange(100):
            hist = None
            x = random.randint(0, self.size_x-1025)
            y = random.randint(0, self.size_y-1025)
            pimg = self.slide.read_region((x,y), self.magn, (1024, 1024))
            pimg = pimg.convert('RGB')
            img = numpy.asarray(pimg)[:,:,[2,1,0]]
            if img.shape[-1] == 1:
                #grayscale
                hist = cv2.calcHist([img], [0], None, [256], [0,255])
            else:
                ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                hist = cv2.calcHist([ycc], [0], None, [256], [0,255])
            csum = 0.0
            for i in xrange(hist.shape[0]):
                csum += hist[i][0]
                hist[i][0] = csum
            #normalise
            hist = hist/csum
            if None is total_hist:
                total_hist = hist
            else:
                total_hist += hist
            total_cnt += 1

        return (total_hist/total_cnt).ravel()

    def get_lcn_params(self):
        total_means = None
        total_stds = None
        total_cnt = 0.0

        for i in xrange(100):
            means, stds = [], []
            x = random.randint(0, self.size_x-1025)
            y = random.randint(0, self.size_y-1025)
            pimg = self.slide.read_region((x,y), self.magn, (1024, 1024))
            pimg = pimg.convert('RGB')
            img = numpy.asarray(pimg)[:,:,[2,1,0]]
            for z in xrange(img.shape[-1]):
                means.append(numpy.mean(img[:,:,z]))
                stds.append(numpy.std(img[:,:,z]))
            total_means = numpy.asarray(means) if None is total_means else total_means + numpy.asarray(means)
            total_stds = numpy.asarray(stds) if None is total_stds else total_stds + numpy.asarray(stds)
            total_cnt += 1

        return (total_means/total_cnt, total_stds/total_cnt)

    def close(self):
        if self.slide:
            self.slide.close()
            self.slide = None
