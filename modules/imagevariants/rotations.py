from scipy import ndimage
from baseimagevariant import BaseImageVariant

class Rotations(BaseImageVariant):

    def __init__(self, logger, config, key_prefix):
        self.logger = logger
        self.config = config
        self.key_prefix = key_prefix

        #load config
        self.label_angles = config.get('rotations', {}).get(key_prefix, {})

    #return an array of tuples (key,variant_img)
    def get_variants(self, imgname, img, label):
        variants = []
        for angle in self.label_angles.get(label, self.label_angles.get('default', [])):
            key = 'ro%d' % angle
            self.logger.debug('adding rotation variant for patch %s with key %s' % (imgname, key))
            variants.append((key, ndimage.rotate(img, angle, reshape=False)))
        return variants
