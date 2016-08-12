import numpy
import cv2
from baseimageproc import BaseImageProc

class Sharpen(BaseImageProc): 

    def __init__(self, logger, config):
        self.logger = logger

    #return tuple (img, inverse_info)
    def process_img(self, imgname, img, img_config={}):
        l = cv2.Laplacian(img, cv2.CV_16S)
        img = img - l
        img = numpy.asarray(numpy.minimum(img, 255), dtype=numpy.uint8)
        return (img, ['sharpen'])
