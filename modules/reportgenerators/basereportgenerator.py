class BaseReportGenerator(object):

    def __init__(self, sys_config, analysis_config, logger, **kwargs):
        pass

    def generate_report(self, report_path):
        raise NotImplementedError('Subclasses must override generate_report')

    def cleanup(self):
        raise NotImplementedError('Subclasses must override cleanup')

def factory(rtype, *args, **kwargs):
    from latexreportgenerator import LatexReportGenerator

    if rtype == 'latex':
        return LatexReportGenerator(*args, **kwargs)
    else:
        raise Exception('unsupported report generator type %s' % rtype)
