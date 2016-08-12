from baseimageproc import BaseImageProc

class GetPlane(BaseImageProc):

    def __init__(self, logger, config):
        self.logger = logger
        self.plane_index = config.get('plane_index', 0)

    #return processed image
    def process_img(self, imgname, img, img_config={}):
        return (img[:,:,self.plane_index], ['getpl:%d' % self.plane_index])
