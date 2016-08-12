import numpy
import cv2
from baseimageproc import BaseImageProc

class ImageConvert(BaseImageProc):

    def __init__(self, logger, config):
        self.logger = logger
        self.conv_str = config.get('image_convert_format', '').lower()
        self.conv_format = None
        if not self.conv_str:
            raise Exception('format not provided for image conversion')
        if self.conv_str == 'bgr2hsv':
            self.conv_format = cv2.COLOR_BGR2HSV
        elif self.conv_str == 'hsv2bgr':
            self.conv_format = cv2.COLOR_HSV2BGR
        elif self.conv_str == 'bgr2ycrcb':
            if hasattr(cv2, 'COLOR_BGR2YCrCb'):
                self.conv_format = cv2.COLOR_BGR2YCrCb
            else:
                self.conv_format = cv2.COLOR_BGR2YCr_CB
        elif self.conv_str == 'ycrcb2bgr':
            if hasattr(cv2, 'COLOR_YCrCb2BGR'):
                self.conv_format = cv2.COLOR_YCrCb2BGR
            else:
                self.conv_format = cv2.COLOR_YCr_CB2BGR
        elif self.conv_str == 'bgr2gray':
            self.conv_format = cv2.COLOR_BGR2GRAY
        elif self.conv_str == 'gray2bgr':
            self.conv_format = cv2.COLOR_GRAY2BGR
        elif self.conv_str == 'bgr2hihsv':
            self.conv_format = cv2.COLOR_BGR2HSV
        elif self.conv_str == 'rgb2gray':
            self.conv_format = cv2.COLOR_RGB2GRAY  
        else:
            raise Exception('unsupport conversion format %s provided' % self.conv_str)

    #return tuple (img, inverse_info)
    def process_img(self, imgname, img, img_config={}):
        processed_img =  cv2.cvtColor(img, self.conv_format)
        if self.conv_str == 'bgr2hihsv':
            processed_img = processed_img + processed_img
        if self.conv_str == 'bgr2gray':
            processed_img = numpy.reshape(processed_img, (img.shape[0], img.shape[1], 1))
        return (processed_img, [self.conv_str])
