class BaseModelPostProc(object):

    def __init__(self, logger, config):
        pass

    #return output path
    def process_model_output(self, model_out_dir):
        raise NotImplementedError('Subclasses must override process_model_output')

def factory(ptype, *args, **kwargs):
    from regionattributes import RegionAttributes
    
    if ptype in ['regionattributes']:
        return RegionAttributes(*args, **kwargs)
    else:
        raise Exception('unsupported model post proc type %s' % ptype)
