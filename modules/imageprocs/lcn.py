import numpy
from baseimageproc import BaseImageProc

class LCN(BaseImageProc): 

    def __init__(self, logger, config):
        self.logger = logger

    #return tuple (img, inverse_info)
    #assumes image planes have been inverted
    def process_img(self, imgname, img, img_config={}):
        means, stds = [], []
        if 'lcn' in img_config:
            (means, stds) = img_config['lcn']
        img = img*1.0
        for z in xrange(img.shape[0]):
            if 'lcn' not in img_config:
                means.append(numpy.mean(img[z,:,:]))
                stds.append(numpy.std(img[z,:,:]))
            img[z,:,:] = (img[z,:,:]-means[z])/stds[z]
        self.logger.debug('lcn for image %s, means: %s, stds: %s' % (imgname, ','.join(map(str, means)), ','.join(map(str, stds))))
        return (img, ['lcnmean:%s' % (','.join(map(str, means))), 'lcnstd:%s' % (','.join(map(str, stds)))])
