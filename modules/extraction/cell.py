#!/usr/bin/env python
''' cell.py: An abstract class for extracting patches of specific types'''

__author__="Rohit Kumar Pandey"
__copyright__="Copyright 2015, SigTuple Technologies Pvt Ltd"
__version__="0.1"
__maintainer__="Rohit Kumar Pandey"
__email__="rohit@sigtuple.com"
__status__="development"

from abc import ABCMeta, abstractmethod

class Cell:

    __metaclass__ = ABCMeta

    #return dict of patch names and images
    @abstractmethod
    def extract_patches(self, input_info, num_patches, all_global_attribs=False): pass

    def get_attribute_key(self, cent_x, cent_y):
        return "{}_{}_{}_{}".format(self.curr_img_name.split(".")[0], self.label, cent_x, cent_y)


def factory(ctype, *args, **kwargs):
    from wbc import Wbc
    from rbc import Rbc

    if ctype == 'rbc':
        return Rbc(*args, **kwargs)
    elif ctype == 'wbc' or ctype == 'platelet':
        return Wbc(*args, **kwargs)
    elif ctype == 'malaria':
        return Wbc(*args, **kwargs)
    elif ctype == 'dummy':
        return Dummy(*arg, **kwargs)    
    else:
        raise Exception('unsupported cell type %s' % ctype)
