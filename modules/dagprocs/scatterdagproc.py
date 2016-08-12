#!/usr/bin/env python

import os,sys
import json
import math
import numpy
import codecs
from basedagproc import BaseDAGProc
from modules.utils.fsutils import DataFile

class ScatterDAGProc(BaseDAGProc):

    def __init__(self, sys_config, config, analysis_config, out_queue, proc_configs, logger=None):
        #invoke base constructor
        super(ScatterDAGProc, self).__init__(sys_config, config, analysis_config, out_queue, proc_configs, logger=logger)

        #validate_config
        self.output_dir = self.config.get('output', None)
        if not self.output_dir:
            raise Exception('no output path provided in config for scatter proc id %s' % self.procid)
        self.output_dir = os.path.join(self.output_dir, self.procid)
        self.output_path = None
        self.output_label = self.config.get('output_label', '')
        if not self.output_label:
            raise Exception('no output label provided for scatter proc id %s' % self.procid)
        self.scatter_name = self.config.get('scatter_name', 'scatter')

        self.name_key = self.config.get('name_key', None)
        self.value_key = self.config.get('value_key', None)
        if not (self.value_key and self.name_key):
            raise Exception('invalid value key or name key provided for scatter proc id %s' % (self.procid))
        self.ignore_axes = set(self.config.get('ignore_axes', []))
        self.scatter_range = self.config.get('scatter_range', [0,255])
        self.scatter_modulo = self.config.get('scatter_modulo', None)
        self.filter_info = self.config.get('filter', {})
        self.per_label_scatter = self.config.get('per_label_scatter', False)
        self.default_label = 'all'
        self.unit = self.config.get('unit', None)
        self.filter_labels = {}
        self.whitelist, self.blacklist = None, None
        self.logger.info('scatter DAG proc id %s initialised successfully' % self.procid)

    def run(self):
        try:
            if self.filter_info:
                self.blacklist = set(self.filter_info.get('blacklist', []))
                self.whitelist = set(self.filter_info.get('whitelist', []))
                file_path = None
                if self.filter_info.get('output_from', ''):
                    if self.filter_info['output_from'] in self.proc_configs:
                        file_path = self.proc_configs.get(self.filter_info['output_from'], {}).get('output_path', None)
                    else:
                        self.logger.error('filter dependency %s not present in done procs, skipping' % self.filter_info['output_from'])
                elif self.filter_info.get('file_path', ''):
                    file_path = self.filter_info.get('file_path', '')
                if file_path and os.path.exists(file_path):
                    self.logger.info('loading filters from %s' % file_path)
                    inf = DataFile(file_path, 'r', self.logger).get_fp()
                    injson = json.load(inf)
                    inputs = injson
                    for k in self.filter_info['input_key'].split(':'):
                        inputs = inputs[k]
                    for label,patches in inputs.iteritems():
                        for info in patches:
                            fpath = info[self.filter_info['name_key']]
                            self.filter_labels[os.path.basename(fpath)] = label
                    inf.close()
    
            input_paths = []
            if 'input' in self.config:
                input_paths = self.config['input']
            elif 'input_from' in self.config:
                ipath = self.proc_configs.get(self.config['input_from'], None).get('output_path', None)
                if ipath:
                    input_paths.append(ipath)
            #handle case where no extraction has happened
            #if len(input_paths) <= 0 or not reduce(lambda x,y: x or y, map(lambda z:os.path.exists(z), input_paths)):
            #    raise Exception('no valid input path provided for scatter proc id %s' % self.procid)
            scatter_axes = []
            scatter_units = None
            scatter_data = {}
            values = {}
            for input_path in input_paths:
                if not input_path or not os.path.exists(input_path):
                    continue
                inf = DataFile(input_path, 'r', self.logger).get_fp()
                for line in inf:
                    fields = json.loads(line.strip())
                    value = fields.get(self.value_key, None)
                    if not value:
                        continue
                    name = os.path.basename(fields[self.name_key])
                    label = self.filter_labels.get(name, '')
                    #check filter
                    if self.blacklist or self.whitelist:
                        if self.whitelist and label not in self.whitelist:
                            continue
                        elif self.blacklist and label in self.blacklist:
                            continue
                    #add to scatter
                    if not scatter_axes:
                        scatter_axes = sorted(value.keys())
                        if self.ignore_axes:
                            scatter_axes = [x for x in scatter_axes if x not in self.ignore_axes]
                        if None is not self.unit:
                            scatter_units = len(scatter_axes)*[self.unit]
                    out_val = []
                    for axis in scatter_axes:
                        ov = value.get(axis, None)
                        values.setdefault(axis,[]).append(ov)
                        if ov and self.scatter_modulo:
                            ov -= ov%self.scatter_modulo
                        out_val.append(ov)
                    out_val = tuple(out_val)
                    scatter_data[out_val] = scatter_data.get(out_val, 0)+1

            # compute standard deviation and mean along all scatter axes
            mean = {}
            std_dev = {}
            for axis in scatter_axes:
                mean[axis] = {"val": numpy.mean(values[axis]), "unit": self.unit}
                std_dev[axis] = {"val": numpy.std(values[axis]),"unit": self.unit}

            #create output
            out_scatter_data = []
            if scatter_axes:
                scatter_axes.append('count')
                if None is not scatter_units:
                    scatter_units.append('')
            for (k,v) in scatter_data.iteritems():
                ov = list(k)
                ov.append(v)
                out_scatter_data.append(ov)
            self.scatter_output_info = {'axes': scatter_axes, 'data': out_scatter_data, 'range': self.scatter_range, 'mean': mean, 'std_dev':std_dev}
            if None is not scatter_units:
                self.scatter_output_info['units'] = scatter_units
            self.output_path = os.path.join(self.output_dir, 'scatter.json')
            self.logger.info('writing output to %s for scatter procid %s' % (self.output_path, self.procid))
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            outf = DataFile(self.output_path, 'w', self.logger).get_fp()
            json.dump({self.output_label: {'scatter': {self.scatter_name: self.scatter_output_info}}}, outf, ensure_ascii=False)
            outf.close()
            if self.out_queue:
                self.out_queue.put({self.procid: {'status': 'success', 'output_path': self.output_path}})
            self.cleanup()
            self.logger.info('scatter run finished for proc_id %s' % self.procid)
        except Exception as e:
            self.logger.exception('exception in processing of scatter proc id %s' % self.procid)
            if self.out_queue:
                self.out_queue.put({self.procid: {'status': 'failure'}})

    def cleanup(self):
        pass
