import cv2
from baseimagevariant import BaseImageVariant

class Reflections(BaseImageVariant):

    def __init__(self, logger, config, key_prefix):
        self.logger = logger
        self.config = config
        self.key_prefix = key_prefix

        #load config
        self.label_axes = config.get('reflections', {}).get(key_prefix, {})

    #return an array of tuples (key,variant_img)
    def get_variants(self, imgname, img, label):
        variants = []
        for axis in self.label_axes.get(label, self.label_axes.get('default', [])):
            key = 'ref%d' % axis
            self.logger.debug('adding reflection variant for patch %s with key %s' % (imgname, key))
            variants.append((key, cv2.flip(img, axis)))
        return variants
