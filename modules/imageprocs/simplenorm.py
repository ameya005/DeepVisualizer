from baseimageproc import BaseImageProc

class SimpleNorm(BaseImageProc):

    def __init__(self, logger, config):
        self.logger = logger
        self.norm_val = config.get('norm_val', 255.)

    def process_img(self, imgname, img, img_config={}):
        self.logger.debug('normalising image %s with parameter %f' % (imgname, self.norm_val))
        return (img/float(self.norm_val), ['snorm:%s' % str(self.norm_val)])
