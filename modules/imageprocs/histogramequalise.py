import numpy
import cv2
from baseimageproc import BaseImageProc

class HistogramEqualise(BaseImageProc): 

    def __init__(self, logger, config):
        self.logger = logger
        self.median_blur_kernel_size = config.get('histeq_blur_kernel_size', -1)

    #return tuple (img, inverse_info)
    def process_img(self, imgname, img, img_config={}):
        conv_flags = None
        out_val = None
        if img.shape[-1] == 1:
            #grayscale image
            out_img = cv2.equalizeHist(img)
            out_val = (out_img.reshape(img.shape), ["histeq"])
        else:
           if hasattr(cv2, 'COLOR_BGR2YCrCb'):
               conv_flag = [cv2.COLOR_BGR2YCrCb, cv2.COLOR_YCrCb2BGR]
           else:
               conv_flag = [cv2.COLOR_BGR2YCr_CB, cv2.COLOR_YCr_CB2BGR]
           ycc = cv2.cvtColor(img, conv_flag[0])
           ycc[:,:,0] = cv2.equalizeHist(ycc[:,:,0])
           if self.median_blur_kernel_size != -1:
               ycc[:,:,0] = cv2.medianBlur(ycc[:,:,0], self.median_blur_kernel_size)
           out_val = (cv2.cvtColor(ycc, conv_flag[1]), ["histeq"])
        return out_val
