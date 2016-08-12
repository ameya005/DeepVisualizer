class BaseImageProc(object):

    def __init__(self, logger, config):
        pass

    #return tuple (processed image, list of inverse info)
    def process_img(self, imgname, img, img_config={}):
        raise NotImplementedError('Subclasses must override process_img')


def factory(ptype, *args, **kwargs):
    from lcn import LCN
    from resize import Resize
    from invertplanes import InvertPlanes
    from invertrgb import InvertRGB
    from simplenorm import SimpleNorm
    from histogramequalise import HistogramEqualise
    from histogrammatch import HistogramMatch
    from imageconvert import ImageConvert
    from medianblur import MedianBlur
    from sharpen import Sharpen
    from normalize import Normalize

    if ptype in ['lcn', 'globallcn']:
        return LCN(*args, **kwargs)
    elif ptype == 'resize':
        return Resize(*args, **kwargs)
    elif ptype == 'getplane':
        return GetPlane(*args, **kwargs)
    elif ptype == 'imageconvert':
        return ImageConvert(*args, **kwargs)
    elif ptype == 'invertplanes':
        return InvertPlanes(*args, **kwargs)
    elif ptype == 'invertrgb':
        return InvertRGB(*args, **kwargs)
    elif ptype == 'simplenorm':
        return SimpleNorm(*args, **kwargs)
    elif ptype == 'histogramequalise':
        return HistogramEqualise(*args, **kwargs)
    elif ptype == 'histogrammatch':
        return HistogramMatch(*args, **kwargs)
    elif ptype == 'medianblur':
        return MedianBlur(*args, **kwargs)
    elif ptype == 'sharpen':
        return Sharpen(*args, **kwargs)
    elif ptype == 'normalize':
        return Normalize(*args, **kwargs)        
    else:
        raise Exception('unsupported image proc type %s' % ptype)
