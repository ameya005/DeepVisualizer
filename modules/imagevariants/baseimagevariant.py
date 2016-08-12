class BaseImageVariant(object):

    def __init__(self, logger, config, key_prefix):
        pass

    #return an array of tuples of (key,variant)
    def get_variants(self, imgname, img, label):
        raise NotImplementedError('Subclasses must override get_variants')

def factory(vtype, *args, **kwargs):
    from rotations import Rotations
    from reflections import Reflections
    from translations import Translations
    from imagequality import ImageQuality
    from contrasts import Contrasts

    if vtype == 'rotations':
        return Rotations(*args, **kwargs)
    elif vtype == 'reflections':
        return Reflections(*args, **kwargs)
    elif vtype == 'translations':
        return Translations(*args, **kwargs)
    elif vtype == 'quality':
        return ImageQuality(*args, **kwargs)
    elif vtype == 'contrasts':
        return Contrasts(*args, **kwargs)    
    else:
        raise Exception('unsupport image variant type %s' % vtype)
