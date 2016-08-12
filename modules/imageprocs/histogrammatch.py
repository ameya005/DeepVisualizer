import numpy
import cv2
from baseimageproc import BaseImageProc

class HistogramMatch(BaseImageProc): 

    def __init__(self, logger, config):
        self.logger = logger
        self.conf_pixmap = config.get('histmatch_pixmap', None)
        self.pixmap = self.conf_pixmap
        def histmap(val):
            return self.pixmap[val]
        self.histmapfunc = numpy.vectorize(histmap)

    #return tuple (img, inverse_info)
    def process_img(self, imgname, img, img_config={}):
        conv_flags = None
        out_val = None
        self.pixmap = img_config.get('histmatch_pixmap', self.conf_pixmap)

        if None is self.pixmap:
            raise Exception('pixmap not provided in config for histogram matching')

        if img.shape[-1] == 1:
            #grayscale image
            out_img = self.histmapfunc(img)
            out_val = (out_img.reshape(img.shape), ["histmatch"])
        else:
            if hasattr(cv2, 'COLOR_BGR2YCrCb'):
                conv_flag = [cv2.COLOR_BGR2YCrCb, cv2.COLOR_YCrCb2BGR]
            else:
                conv_flag = [cv2.COLOR_BGR2YCr_CB, cv2.COLOR_YCr_CB2BGR]
            ycc = cv2.cvtColor(img, conv_flag[0])
            ycc[:,:,0] = self.histmapfunc(ycc[:,:,0])
            out_val = (cv2.cvtColor(ycc, conv_flag[1]), ["histmatch"])
        return out_val
