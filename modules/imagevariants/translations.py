import cv2
import numpy
from baseimagevariant import BaseImageVariant

class Translations(BaseImageVariant):

    def __init__(self, logger, config, key_prefix):
        self.logger = logger
        self.config = config
        self.key_prefix = key_prefix

        #load config
        self.offsets = {}
        #setup parameters
        newoffsets = {}
        for k,v in config.get('translations', {}).get(key_prefix, {}).iteritems():
            newv = []
            for (xoffset,yoffset) in v:
                newv.append(('%d.%d' % (xoffset, yoffset),numpy.float32([[1,0,xoffset],[0,1,yoffset]])))
            self.offsets[k] = newv

    #return an array of tuples (key,variant_img)
    def get_variants(self, imgname, img, label):
        variants = []
        for (kval, trans_mat) in self.offsets.get(label, self.offsets.get('default', [])):
            key = 'tr%s' % kval
            self.logger.debug('adding translation variant for patch %s with key %s' % (imgname, key))
            variants.append((key, cv2.warpAffine(img, trans_mat, (img.shape[1], img.shape[0]))))
        return variants
