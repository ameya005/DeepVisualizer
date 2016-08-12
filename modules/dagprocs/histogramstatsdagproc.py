#!/usr/bin/env python

import os,sys
import json
import math
import numpy
import codecs
from basedagproc import BaseDAGProc
from modules.utils.fsutils import DataFile

class HistogramStatsDAGProc(BaseDAGProc):

    def __init__(self, sys_config, config, analysis_config, out_queue, proc_configs, logger=None):
        #invoke base constructor
        super(HistogramStatsDAGProc, self).__init__(sys_config, config, analysis_config, out_queue, proc_configs, logger=logger)

        #validate_config
        self.is_histogram = (self.config.get('type', 'stats') == 'histogram')
        self.output_dir = self.config.get('output', None)
        if not self.output_dir:
            raise Exception('no output path provided in config for histogram stats proc id %s' % self.procid)
        self.output_dir = os.path.join(self.output_dir, self.procid)
        self.output_path = None
        self.output_label = self.config.get('output_label', '')
        if not self.output_label:
            raise Exception('no output label provided for histogram stats proc id %s' % self.procid)
        self.output_name = self.config.get('output_name', 'histogram')

        self.name_key = self.config.get('name_key', None)
        self.value_key = self.config.get('value_key', None)
        if not (self.value_key and self.name_key):
            raise Exception('invalid value key or name key provided for histogram stats proc id %s' % (self.procid))
        self.ignore_values = set(self.config.get('ignore_values', []))
       
        if self.is_histogram:
            self.hist_range = self.config.get('range', [])
            if not self.hist_range or len(self.hist_range) < 2:
                raise Exception('invalid range provided for histogram stats proc id %s' % self.procid)
            self.bucket_size = float(self.config.get('bucket_size', -1.))
            if self.bucket_size <= 0:
                self.bucket_size = (self.hist_range[1]-self.hist_range[0])/10.
            self.cumulative = self.config.get('cumulative', False)
            self.histogram = {}
            self.global_histogram = self.config.get('global_histogram', {})
        self.per_label = self.config.get('per_label', False)
        self.num_output_patches = self.config.get("num_output_patches", 0)
        self.hist_patches = {} if self.num_output_patches else None
        self.stats = {}
        self.filter_info = self.config.get('filter', {})
        self.default_label = 'all'
        self.filter_labels = {}
        self.unit = self.config.get('unit', None)
        self.whitelist, self.blacklist = None, None
        self.logger.info('histogram stats DAG proc id %s initialised successfully' % self.procid)

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
                            self.filter_labels[os.path.basename(fpath)] = {'label':label, 'prob': info['prob'], 'orig_path': fpath, 'name':os.path.basename(fpath)}
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
            #    raise Exception('no valid input path provided for histogram proc id %s' % self.procid)

            for input_path in input_paths:
                if not input_path:
                    continue
                try:
                    inf = DataFile(input_path, 'r', self.logger).get_fp()
                except:
                    self.logger.info("File %s not found on disk and remote location" % input_path)
                    continue
                for line in inf:
                    fields = json.loads(line.strip())
                    value = float(fields[self.value_key])
                    #check ignore
                    if value in self.ignore_values:
                        continue
                    name = os.path.basename(fields[self.name_key])
                    label = self.filter_labels.get(name, {}).get('label','')
                    #check filter
                    if self.blacklist or self.whitelist:
                        if self.whitelist and label not in self.whitelist:
                            continue
                        elif self.blacklist and label in self.blacklist:
                            continue
                    #add to histogram and stats
                    hists = []
                    stats = []
                    if self.per_label:
                        if self.is_histogram:
                            hists.append(self.histogram.setdefault(label, [0]*(2+int(math.ceil((self.hist_range[1]-self.hist_range[0])/self.bucket_size)))))
                        stats.append(self.stats.setdefault(label, []))
                    #default labels
                    if self.is_histogram:
                        hists.append(self.histogram.setdefault(self.default_label, [0]*(2+int(math.ceil((self.hist_range[1]-self.hist_range[0])/self.bucket_size)))))
                    stats.append(self.stats.setdefault(self.default_label, []))
                    for stat in stats:
                        stat.append(value)
                    if self.is_histogram:
                        for hist in hists:
                            if value < self.hist_range[0]:
                                ind = 0
                            elif value > self.hist_range[1]:
                                ind = -1
                            else:
                                ind = 1+int(math.floor((value-self.hist_range[0])/self.bucket_size))
                            self.logger.debug('adding %f to %d' % (value, ind))
                            hist[ind] = hist[ind]+1
                            if self.hist_patches != None and hist[ind] <= self.num_output_patches:
                                self.logger.debug("Adding patch %s to output patches" % name)
                                self.hist_patches[name] = self.filter_labels.get(name,{})
                                self.hist_patches[name][self.output_name] = value

            #create output
            self.hist_output_info = {}
            self.stats_output_info = {}
            for key,values in self.stats.iteritems():
                if not self.per_label and key != self.default_label:
                    continue
                values = numpy.array(values)
                cnt = len(values)
                mean = round(numpy.mean(values), 2)
                std = round(numpy.std(values), 2)
                #calculate pctiles
                pctile = list(numpy.percentile(values, list(numpy.arange(101, dtype=numpy.float32))))
                if None is not self.unit:
                    pctile = [{'val': z, 'unit': self.unit} for z in pctile]
                    self.stats_output_info[key] = {'count': {'val': cnt}, 'mean': {'val': mean, 'unit': self.unit}, 'std': {'val': std, 'unit': self.unit}, 'pctile': pctile}
                else:
                    self.stats_output_info[key] = {'count': cnt, 'mean': mean, 'std': std, 'pctile': pctile}


            if self.is_histogram:
                for key,hist in self.histogram.iteritems():
                    if not self.per_label and key != self.default_label:
                        continue
                    hist_output_info = []
                    hist_output_info.append(['< %0.2f' % self.hist_range[0], hist[0]])
                    for i in xrange(1, len(hist)-1):
                        minval = self.hist_range[0]+(i-1)*self.bucket_size
                        maxval = minval+self.bucket_size
                        hist_output_info.append(['%0.2f - %0.2f' % (minval, maxval), hist[i]])
                    hist_output_info.append(['> %0.2f' % self.hist_range[1], hist[-1]])
                    #normalise histogram values
                    if None is not self.unit:
                        if self.stats_output_info[key]['count']['val'] > 0:
                            for i in xrange(len(hist_output_info)):
                                hist_output_info[i][1] = (hist_output_info[i][1]*1.0)/self.stats_output_info[key]['count']['val']
                    else:
                        if self.stats_output_info[key]['count'] > 0:
                            for i in xrange(len(hist_output_info)):
                                hist_output_info[i][1] = (hist_output_info[i][1]*1.0)/self.stats_output_info[key]['count']
                    self.hist_output_info[key] = {'hist': hist_output_info}
                    if None is not self.unit:
                        self.hist_output_info[key]['unit'] = self.unit
                    #add global histogram if present
                    if self.cumulative:
                        self.hist_output_info[key]['cumulative'] = []
                        cumulative_count = 0.0
                        for i in xrange(len(self.hist_output_info[key]['hist'])):
                            cumulative_count += self.hist_output_info[key]['hist'][i][1]
                            self.hist_output_info[key]['cumulative'].append([self.hist_output_info[key]['hist'][i][0], cumulative_count])
                    if key in self.global_histogram:
                        self.hist_output_info[key]['global'] = self.global_histogram[key]

            self.output_path = os.path.join(self.output_dir, 'statsnhistogram.json')
            self.logger.info('writing output to %s for histogram stats procid %s' % (self.output_path, self.procid))
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            outf = DataFile(self.output_path, 'w', self.logger).get_fp()
            out_json = {self.output_label: {'stats': {self.output_name: self.stats_output_info}}}
            if self.is_histogram:
                out_json[self.output_label]['histogram'] = {self.output_name: self.hist_output_info}
            if self.hist_patches != None:
                out_json[self.output_label]['output_patch'] = self.hist_patches
            json.dump(out_json, outf, ensure_ascii=False)
            outf.close()
            if self.out_queue:
                self.out_queue.put({self.procid: {'status': 'success', 'output_path': self.output_path}})
            self.cleanup()
            self.logger.info('histogram run finished for proc_id %s' % self.procid)
        except Exception as e:
            self.logger.exception('exception in processing of histogram stats proc id %s' % self.procid)
            if self.out_queue:
                self.out_queue.put({self.procid: {'status': 'failure'}})

    def cleanup(self):
        pass
