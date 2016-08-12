import numpy
from baseimageproc import BaseImageProc

class InvertRGB(BaseImageProc): 

    def __init__(self, logger, config):
        self.logger = logger

    #return tuple (img, inverse_info)
    #assumes 2D image
    def process_img(self, imgname, img, img_config={}):
        self.logger.debug('inverting RGB for image %s' % (imgname))
        img = img[:,:,[2,1,0]]
        return (img, ['irgb'])
