import cv2
import numpy as np
from baseimagevariant import BaseImageVariant

class Contrasts(BaseImageVariant):


    def __init__(self, logger, config, key_prefix):
        self.logger = logger
        self.config = config
        self.key_prefix = key_prefix

        #load config
        self.gammas = {}
        #setup parameters
        self.gammas = config.get('contrasts', {}).get(key_prefix, {})
        

    #return an array of tuples (key,variant_img)
    def get_variants(self, imgname, img, label):
        variants = []
        for gamma in self.gammas.get(label, self.gammas.get('default', [])):
            key = 'co%d' % (gamma*10)
            self.logger.debug('adding contrast variant for patch %s with key %s' % (imgname, key))
            variants.append((key, self.adjust_gamma(img,gamma)))
        return variants

    #adjsuts the brightness/contast_value as per user input    
    def adjust_gamma(self, image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        self.logger.debug('Changing contrast/Brightness with gamma value:%f', gamma)
        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)    