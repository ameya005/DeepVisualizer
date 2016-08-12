import os
import math
import baseinputreader
import numpy
import cv2
from modules.utils.fsutils import DataFile

class ImageReader(baseinputreader.BaseInputReader):
    def __init__(self, logger, inpath, jump=1024, magn=0, patch_size=1024, in_ppm=(1.0,1.0), out_ppm=(1.0,1.0)):
        self.logger = logger
        self.inpath = inpath

        #fetch from remote data store if reqd
        infp = DataFile(inpath, 'r', self.logger).get_fp()

        self.image = cv2.imread(inpath.encode('utf-8'))

        infp.close()
        if None is self.image or not self.image.any():
            raise Exception('could not open image at %s' % inpath)

        self.in_ppm = in_ppm
        self.out_ppm = out_ppm
        self.scale_x = self.out_ppm[0]/self.in_ppm[0]
        self.scale_y = self.out_ppm[1]/self.in_ppm[1]

        self.jump_x = jump
        self.jump_y = jump
        self.patch_size_x = patch_size
        self.patch_size_y = patch_size
        self.ind_x = 0
        self.ind_y = 0
        #only supporting 0 magn for now
        if magn != 0:
            raise Exception('unsupport magnification %d for slide %s' % (magn, inpath))
        self.magn = magn
        self.image = cv2.resize(self.image, (0, 0), fx=self.scale_x, fy=self.scale_y)
        self.size_y, self.size_x = self.image.shape[0], self.image.shape[1]
        self.logger.debug('size of image %s' % (str(self.image.shape)))

    def get_type(self):
        return 'image'

    def get_properties(self):
        props = {}

        props["res_x"] = self.image.shape[1]
        props["res_y"] = self.image.shape[0]
        return props

    def get_name(self):
        return os.path.basename(self.inpath)

    def get_size(self):
        return (self.size_x, self.size_y)

    def next(self):
        #get next patch
        if self.ind_y >= self.size_y:
            self.ind_y = 0
            self.ind_x += self.jump_x
        if self.ind_x >= self.size_x:
            raise StopIteration
        ret_img = self.image[self.ind_y:self.ind_y+self.patch_size_y,self.ind_x:self.ind_x+self.patch_size_x]
        out_x, out_y = self.ind_x, self.ind_y
        self.ind_y += self.jump_y
        #pad if req
        if ret_img.shape[0] < self.patch_size_y or ret_img.shape[1] < self.patch_size_x:
            ret_img = numpy.pad(ret_img, ((0,self.patch_size_y-ret_img.shape[0]),(0,self.patch_size_x-ret_img.shape[1]),(0,0)), \
                                mode='constant', constant_values=(0,))
        self.logger.debug('size of iter patch: %s' % (str(ret_img.shape)))
        return (out_x, out_y, ret_img)

    def reset(self, jump=-1, patch_size=-1, magn=-1, out_ppm=None):
        self.ind_x = 0
        self.ind_y = 0
        if jump > 0:
            self.logger.info('changing jump value to %d' % (jump))
            self.jump_x,self.jump_y = jump,jump
        if patch_size > 0:
            self.logger.info('changing patch_size value to %d' % (patch_size))
            self.patch_size_x,self.patch_size_y = patch_size,patch_size
        if magn != -1 and magn != self.magn:
            if magn != 0:
                raise Exception('unsupport magnification %d' % (magn))
            self.logger.info('Changing magnification value from %d to %d' % (self.magn, magn))
            self.magn = magn
        if out_ppm:
            self.out_ppm = out_ppm
            self.scale_x = self.out_ppm[0]/self.in_ppm[0]
            self.scale_y = self.out_ppm[1]/self.in_ppm[1]
            self.image = cv2.imread(self.inpath.encode('utf-8'))
            self.image = cv2.resize(self.image, (0, 0), fx=self.scale_x, fy=self.scale_y)
            self.size_y, self.size_x = self.image.shape[0], self.image.shape[1]

    def get_patch(self, x, y, size_x, size_y):
        if x > self.size_x or y > self.size_y:
            raise Exception('provided coordinates (%d,%d) lie outside the image size (%d,%d)' % (x,y,self.size_x,self.size_y))
        ret_img = numpy.zeros((size_x, size_y, 3), dtype=self.image.dtype)
        x1,x2 = max(0,x), min(x+size_x, self.size_x)
        y1,y2 = max(0,y), min(y+size_y, self.size_y)
        ret_img[max(0,0-y):(y2-y1)+max(0,0-y),max(0,0-x):(x2-x1)+max(0,0-x)] = self.image[y1:y2,x1:x2]
        #if ret_img.shape[0] < size_y or ret_img.shape[1] < size_x:
        #    ret_img = numpy.pad(ret_img, ((0,size_y-ret_img.shape[0]),(0,size_x-ret_img.shape[1]),(0,0)), \
        #                mode='constant', constant_values=(0,))
        self.logger.debug('size of patch: %s' % (str(ret_img.shape)))
        return ret_img

    def get_cumulative_histogram(self):
        hist = None
        if self.image.shape[-1] == 1:
            #grayscale
            hist = cv2.calcHist([self.image], [0], None, [256], [0,255])
        else:
            ycc = cv2.cvtColor(self.image, cv2.COLOR_BGR2YCrCb)
            hist = cv2.calcHist([ycc], [0], None, [256], [0,255])
        csum = 0.0
        for i in xrange(hist.shape[0]):
            csum += hist[i][0]
            hist[i][0] = csum
        #normalise
        hist = hist/csum
        return hist.ravel()

    def get_lcn_params(self):
        means, stds = [], []
        for z in xrange(self.image.shape[-1]):
            means.append(numpy.mean(self.image[:,:,z]))
            stds.append(numpy.std(self.image[:,:,z]))
        return (numpy.asarray(means), numpy.asarray(stds))

    def close(self):
        if None is not self.image and self.image.any():
            del self.image
            self.image = None
