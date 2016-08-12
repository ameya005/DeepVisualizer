import numpy
import cv2
from baseimageproc import BaseImageProc

class MedianBlur(BaseImageProc): 

    def __init__(self, logger, config):
        self.logger = logger
        self.filter_size = config.get('median_filter_size', 3)
        self.filter_plane = config.get('median_filter_plane', 0)

    #return tuple (img, inverse_info)
    def process_img(self, imgname, img, img_config={}):
        img[:,:,self.filter_plane] = cv2.medianBlue(img[:,:,self.filter_plane], self.filter_size)
        return (img, ['medianblur:%d:%d' % (self.filter_size, self.filter_plane)])
