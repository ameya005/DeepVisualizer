import os
import math
import baseinputreader
import numpy
import cv2
from modules.utils.fsutils import DataFile

#this class does nothing...yet
class GenericFileReader(baseinputreader.BaseInputReader):
    def __init__(self, logger, inpath, jump=1024, magn=0, patch_size=1024, in_ppm=(1.0,1.0), out_ppm=(1.0,1.0)):
        self.logger = logger
        self.inpath = inpath

        #fetch from remote data store if reqd
        infp = DataFile(inpath, 'r', self.logger).get_fp()
        infp.close()
        self.size = os.path.getsize(self.inpath)
        self.in_ppm = in_ppm
        self.out_ppm = out_ppm

    def get_type(self):
        return 'genericfile'

    def get_properties(self):
        props = {}

        props['size'] = self.size
        
        return props

    def get_name(self):
        return os.path.basename(self.inpath)

    def get_size(self):
        return (self.size, self.size)

    def next(self):
        raise StopIteration

    def reset(self, jump=-1, patch_size=-1, magn=-1, out_ppm=None):
        pass

    def get_patch(self, x, y, size_x, size_y):
        return None

    def get_cumulative_histogram(self):
        return None

    def get_lcn_params(self):
        return None

    def close(self):
        pass
