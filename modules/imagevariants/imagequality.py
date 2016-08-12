import cv2
import numpy
import random
from baseimagevariant import BaseImageVariant

class ImageQuality(BaseImageVariant):

    def __init__(self, logger, config, key_prefix):
        self.logger = logger
        self.config = config
        self.key_prefix = key_prefix

        #load config
        self.qinfo = config.get('quality', {}).get(key_prefix, {})

    #return an array of tuples (key,variant_img)
    def get_variants(self, imgname, img, label):
        variants = []
        info = self.qinfo.get(label, self.qinfo.get('default', {}))
        if random.random() < info.get('prob', 0.0):
            for qlevel in info.get('levels', []):
                key = 'qu%d' % qlevel
                retval,qbuf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, qlevel])
                imread_flag = 1
                if len(img.shape) == 2 or img.shape[-1] == 1:
                    imread_flag = 0
                self.logger.debug('adding image quality variant for patch %s with key %s and imread flag %d' % (imgname, key, imread_flag))
                variants.append((key, cv2.imdecode(qbuf, imread_flag)))
        return variants
