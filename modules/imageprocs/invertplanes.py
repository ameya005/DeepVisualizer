import numpy
from baseimageproc import BaseImageProc

class InvertPlanes(BaseImageProc): 

    def __init__(self, logger, config):
        self.logger = logger

    #return tuple (img, inverse_info)
    #assumes 2D image
    def process_img(self, imgname, img, img_config={}):
        self.logger.debug('inverting planes for img %s' % (imgname))
        img = numpy.swapaxes(img, 1, 2)
        img = numpy.swapaxes(img, 0, 1)
        return (img, ['ip'])
