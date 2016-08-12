import os
from datetime import datetime
import multiprocessing
from modules.utils.logger import Logger

class BaseDAGProc(multiprocessing.Process):

    def __init__(self, sys_config, config, analysis_config, out_queue, proc_configs, logger=None, **kwargs):
        #initialise multiprocessing
        super(BaseDAGProc, self).__init__()

        self.analysis_config = analysis_config
        self.sys_config = sys_config
        self.config = config
        self.procid = self.config.get('id', None)
        self.name = '%s_%s' % (self.analysis_config.get('analysis_id', ''), self.procid)
        self.out_queue = out_queue
        self.proc_configs = proc_configs
        if not self.procid:
            raise Exception('no id present in DAG proc config')
        #init logger
        if logger:
            self.logger = logger
        else:
            self.logger = Logger(self.name, os.path.join(self.sys_config['paths']['log'], 'dag_%s_%s.log' % (self.name, datetime.utcnow().strftime('%Y%m%dT%H%M%S'))), 'info', False).get_logger()

    def get_id(self):
        return self.procid

    def run(self):
        raise NotImplementedError('Subclasses must override run')

    def get_output_path(self):
        raise NotImplementedError('Subclasses must override get_output_path')

    def cleanup(self):
        raise NotImplementedError('Subclasses must override cleanup')

    def recompute(self):
        raise NotImplementedError('Subclasses must override recompute')

def factory(ptype, *args, **kwargs):
    from modelinvokerdagproc import ModelInvokerDAGProc
    from modelcollatordagproc import ModelCollatorDAGProc
    from histogramstatsdagproc import HistogramStatsDAGProc
    from scatterdagproc import ScatterDAGProc
    from outputdagproc import OutputDAGProc

    if ptype == 'model':
        return ModelInvokerDAGProc(*args, **kwargs)
    elif ptype == 'model_collator':
        return ModelCollatorDAGProc(*args, **kwargs)
    elif ptype == 'histogram' or ptype == 'stats':
        return HistogramStatsDAGProc(*args, **kwargs)
    elif ptype == 'output':
        return OutputDAGProc(*args, **kwargs)
    elif ptype == 'scatter':
        return ScatterDAGProc(*args, **kwargs)
    else:
        raise Exception('unsupported DAG proc type %s' % ptype)
