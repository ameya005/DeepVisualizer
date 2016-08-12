import numpy as np
import cv2
from baseimageproc import BaseImageProc

class Normalize(BaseImageProc): 

    def __init__(self, logger, config):
        self.logger = logger
        #self.filter_plane = config.get('median_filter_plane', 0)

    #return tuple (img, inverse_info)
    def process_img(self, imgname, img, img_config={}):
        #img[:,:,self.filter_plane] = cv2.medianBlue(img[:,:,self.filter_plane], self.filter_size)
        self.mean = np.mean(img)
        self.std = np.std(img) + 0.00001
        img = img - self.mean
        img = img /  ( self.std )
        return (img, ['normalize:%d:%d' % ((self.mean*10), (self.std*10))])