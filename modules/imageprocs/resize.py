from baseimageproc import BaseImageProc

class Resize(BaseImageProc):

    def __init__(self, logger, config):
        self.logger = logger
        self.out_ppm = config.get('ppm', [])
        self.output_size = config.get('output_size_um', 0.0)
        if not self.out_ppm or not self.output_size:
            raise Exception('mandatory parameters ppm and output_size_um not provided in config')
        #convert output_size to pixels
        self.output_size = [int(self.output_size*self.out_ppm[0]), int(self.output_size*self.out_ppm[1])]

    #return processed image
    def process_img(self, imgname, img, img_config={}):
        self.logger.debug('resizing img %s from (%d,%d) to (%d,%d)' % (imgname, img.shape[1], img.shape[0], self.output_size[0], self.output_size[1]))
        if img.shape[0] == self.output_size[1] and img.shape[1] == self.output_size[0]:
            return (img, [])
        elif img.shape[0] < self.output_size[1] or img.shape[1] < self.output_size[0]:
            self.logger.error('image size (%d,%d) less than required output size (%d,%d)' %  (img.shape[0], img.shape[1], self.output_size, self.output_size))
            return (None, [])
        else:
            x1,y1 = (img.shape[0]-self.output_size[1])/2,(img.shape[1]-self.output_size[0])/2
            x2,y2 = x1+self.output_size[1],y1+self.output_size[0]
            return (img[x1:x2,y1:y2], [])
