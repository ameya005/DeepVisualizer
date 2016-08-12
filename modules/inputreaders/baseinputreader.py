class BaseInputReader(object):

    def __init__(self, logger, inpath, **kwargs):
        pass

    def __iter__(self):
        return self

    def get_type(self):
        raise NotImplementedError('Subclasses must override get_type')

    def get_properties(self):
        raise NotImplementedError('Subclasses must override get_properties')

    def get_name(self):
        raise NotImplementedError('Subclasses must override get_name')

    def get_size(self):
        raise NotImplementedError('Subclasses must override get_size')
        
    def next(self):
        raise NotImplementedError('Subclasses must override next')

    def reset(self, **kwargs):
        raise NotImplementedError('Subclasses must override reset')

    def get_patch(self, *args, **kwargs):
        raise NotImplementedError('Subclasses must override get_patch')

    def get_cumulative_histogram(self, *args, **kwargs):
        raise NotImplementedError('Subclasses must override get_cumulative_histogram')

    def get_lcn_params(self, *args, **kwargs):
        raise NotImplementedError('Subclasses must override get_lcn_params')

    def close(self):
        raise NotImplementedError('Subclasses must override close')

def factory(rtype, *args, **kwargs):
    from slidereader import SlideReader
    from imagereader import ImageReader
    from genericfilereader import GenericFileReader

    if rtype == 'slide':
        return SlideReader(*args, **kwargs)
    elif rtype == 'image':
        return ImageReader(*args, **kwargs)
    else:
        return GenericFileReader(*args, **kwargs)
        #raise Exception('unsupported input reader type %s' % rtype)
