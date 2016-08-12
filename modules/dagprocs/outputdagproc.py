#!/usr/bin/env python

import os,sys
import json
from basedagproc import BaseDAGProc
from modules.reportgenerators import basereportgenerator
from modules.utils.fsutils import DataFile
import numpy


class OutputDAGProc(BaseDAGProc):
    @staticmethod
    def merge_dicts(dest, src):
        if type(src) != dict:
            return
        for k,v in src.iteritems():
            if k in dest:
                vdest = dest[k]
                OutputDAGProc.merge_dicts(vdest, v)
            else:
                dest[k] = v

    @staticmethod
    def get_compound_key(data, key):
        if key == 'constant':
            return 1.0

        keys = key.split(':')
        kdata = data

        for k in keys:
            if None is kdata:
                break
            if type(kdata) is list:
                kdata = kdata[int(k)]
            else:
                kdata = kdata.get(k, None)

        return kdata

    @staticmethod
    def get_op_value(data, opval, val):
        if opval == "VALUE":
            return val
        elif type(opval) is str or type(opval) is unicode:
            return OutputDAGProc.get_compound_key(data, opval)
        else:
            return opval

    @staticmethod
    def calc(data, calc_info, val):
        for cinfo in calc_info:
            for opcode,opvals in cinfo.iteritems():
                if opcode == 'round':
                    val = round(val, opvals[0])
                elif opcode == 'cast':
                    if opvals[0] == 'int':
                        val = int(val)
                    elif opvals[1] == 'float':
                        val = float(val)
                    elif opvals[1] == 'unicode':
                        val = unicode(val)
                    elif opvals[1] == 'str':
                        val = str(val)
                else:
                    v1 = OutputDAGProc.get_op_value(data, opvals[0], val)
                    v2 = OutputDAGProc.get_op_value(data, opvals[1], val)
                    if None is v1 or None is v2:
                        val = 0.0
                    elif opcode == 'add':
                        val = v1 + v2
                    elif opcode == 'subtract':
                        val = v1 - v2
                    elif opcode == 'multiply':
                        val = v1 * v2
                    elif opcode == 'divide':
                        if None is v2 or v2 == 0.0:
                            val = None
                        else:
                            val = v1 / v2
        return val

    @staticmethod
    def compute_stats(data, input, mandatory_keys, precalc):
        values = []
        for file_name, attributes in input.iteritems():
            if all(key in attributes for key in mandatory_keys):
                vval = OutputDAGProc.calc(attributes, precalc, None)
                if None is not vval:
                    values.append(vval)
        if len(values) == 0:
            return {'count': 0, 'mean': 0, 'std': 0, 'pctile': [0]*101}
        values = numpy.array(values)
        cnt = len(values)
        mean = numpy.mean(values)
        std = numpy.std(values)
        percentiles = list(numpy.percentile(values, list(numpy.arange(101, dtype=numpy.float32))))
        stats = {'count': cnt, 'mean': mean, 'std': std, 'pctile': percentiles}
        return stats

            
    def __init__(self, sys_config, config, analysis_config, out_queue, proc_configs, logger=None):
        #invoke base constructor
        super(OutputDAGProc, self).__init__(sys_config, config, analysis_config, out_queue, proc_configs, logger=logger)

        #validate_config
        self.input_paths = self.config.get('input', [])
        self.input_base_keys = self.config.get('input_merge_base_key', '').split(':')


        self.output_dir = self.config.get('output', None)
        if not self.output_dir:
            raise Exception('no output path provided in config for DAG proc id %s' % self.procid)
        self.output_dir = os.path.join(self.output_dir, self.procid)
        self.output_path = None

        #ignore fields for report 
        self.report_ignore_fields = set(self.config.get('report_ignore_fields', []))
        #create report generator
        self.report_generator = None
        if self.config.get('report_type', ''):
            self.report_generator = basereportgenerator.factory(self.config.get('report_type', ''), self.sys_config, self.analysis_config, self.logger)
        self.logger.info('output DAG proc id %s initialised successfully' % self.procid)

    def remove_ignore_fields(self, data):
        for f in self.report_ignore_fields:
            if f in data:
                del data[f]

        for k,v in data.iteritems():
            if type(v) is dict:
                self.remove_ignore_fields(v)

    def add_derived_variable(self, key, varinfo):
        #assume float
        val = 0.0

        precalc = varinfo.get('precalculate', None)
        postcalc = varinfo.get('postcalculate', None)

        if varinfo['type'] == "stats":
            input = OutputDAGProc.get_compound_key(self.output_json, varinfo.get("input", None))
            val = OutputDAGProc.compute_stats(self.output_json, input, varinfo.get("mandatory_keys", []), precalc)
        elif varinfo['type'] == 'lincom':
            #add coeff*value for each in varinfo
            for vkey,vcoeff in varinfo.get('coefficients', {}).iteritems():
                vval = OutputDAGProc.get_compound_key(self.output_json, vkey)
                if None is not precalc and vkey != 'constant':
                    vval = OutputDAGProc.calc(self.output_json, precalc, vval)
                if None is not vval:
                    val += vcoeff*vval
        
        #postcalculate
        if None is not postcalc:
            val = OutputDAGProc.calc(self.output_json, postcalc, val)

        #display check
        display_info = None
        display_check_info = varinfo.get('display_check', {})
        if display_check_info:
            if 'lb' in display_check_info and val < display_check_info['lb']:
                display_info = 'less than %s' % str(display_check_info['lb'])
            elif 'ub' in display_check_info and val > display_check_info['ub']:
                display_info = 'greater than %s' % str(display_check_info['ub'])
    
        #add final value to json
        vjson = self.output_json
        keys = key.split(':')
        for k in keys[:-1]:
            vjson = vjson.setdefault(k, {})
        if 'unit' in varinfo:
            kdata = vjson.setdefault(keys[-1], {})
            kdata['val'] = val
            kdata['unit'] = varinfo['unit']
            if None is not display_info:
                kdata['display'] = display_info
        elif None is not display_info:
            kdata = vjson.setdefault(keys[-1], {})
            kdata['val'] = val
            kdata['display'] = display_info
        else:
            vjson[keys[-1]] = val

    def run(self):
        try:
            self.output_json = {}
            for pid in self.config.get('depends', []):
                inpath = self.proc_configs[pid].get('output_path', None)
                if not inpath or not os.path.exists(inpath):
                    raise Exception('output of proc %s not available at %s' % (pid, inpath))
                self.logger.info('loading output of proc %s from %s' % (pid, inpath))
                inf = DataFile(inpath, 'r', self.logger).get_fp()
                injson = json.load(inf)
                inf.close()
                for label, data in injson.iteritems():
                    label_data = self.output_json.setdefault(label, {})
                    OutputDAGProc.merge_dicts(label_data, data)

            #merge static report data
            if 'static_report_data' in self.config:
                OutputDAGProc.merge_dicts(self.output_json, self.config.get('static_report_data', {}))

            #add info to output json
            self.output_json['analysis'] = {}
            self.output_json['analysis']['num_input'] = self.analysis_config.get('num_input', 0)
            self.output_json['analysis']['sample_id'] = self.analysis_config.get('sample_id', 0)

            #merge inputs with output json
            input_merge_base = self.output_json
            for k in self.input_base_keys:
                if not k:
                    continue
                input_merge_base = input_merge_base.setdefault(k, {})

            for infile in self.input_paths:
                if not infile:
                    continue
                try:
                    inf = DataFile(infile, 'r', self.logger).get_fp()
                except:
                    self.logger.info("File %s not found on disk and remote location" % infile)
                    continue
                #assumption: each line is a JSON
                for line in inf:
                    if not line or not line.strip():
                        continue
                    injson = json.loads(line.strip())
                    for k, v in injson.iteritems():
                        key_data = input_merge_base.setdefault(k, {})
                        OutputDAGProc.merge_dicts(key_data, v)




            #calculate derived variables
            for variable in self.config.get('derived', []):
                self.add_derived_variable(variable['outkey'], variable)

            # Add suggestions to output json
            # If the keys used here are not present in output json, it will fail
            suggestions = {}
            for suggestion, suggestion_exprs in self.config.get("suggestions", {}).iteritems():
                for index, variable in enumerate(suggestion_exprs['variables'], start=1):
                    for expression in suggestion_exprs['exprs']:
                        expression['expr'] = expression['expr'].replace("$%d" % index, str(self.get_compound_key(self.output_json, variable)))
                        for key in expression['reason'].keys():
                            if type(expression['reason'][key]) is dict and ("$%d" % index) == expression['reason'][key].get('value', None):
                                expression['reason'][key]['value'] = float(expression['reason'][key]['value'].replace("$%d" % index, str(self.get_compound_key(self.output_json, variable))))
                for expression in suggestion_exprs['exprs']:
                    self.logger.debug("evaluating expression for suggestion %s: %s" % (suggestion, expression['expr']))
                    exp_eval = False
                    try:
                        exp_eval = eval(expression['expr'])
                    except:
                        self.logger.info('exception during eval of expression: %s, ignoring' % expression['expr'])
                        continue
                    if exp_eval:
                        suggestions[suggestion] = expression['reason']
                        break

            self.output_json['suggestions'] = suggestions

            #serialise output json
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            output_json_path = os.path.join(self.output_dir, 'output.json')
            self.logger.info('proc %s writing output to %s' % (pid, self.output_dir))
            outf = DataFile(output_json_path, 'w', self.logger).get_fp()
            json.dump(self.output_json, outf, ensure_ascii=False)
            outf.close()
            #create report
            if self.report_generator:
                self.output_path = self.report_generator.generate_report(output_json_path, self.output_dir)
            else:
                self.output_path = output_json_path

            #Create output patches JSON
            if self.config.get('output_patches', None):
                output_patches_json_path = None
                output_patches_json = []
                label_subtypes = {}
                all_patches = {}
                for label, label_config in self.config['output_patches'].iteritems():
                    label_subtypes.setdefault(label, [])
                    ignore_labels = label_config.get('ignore_labels', [])

                    num_of_patches = None
                    if label_config.get('num', None):
                        num_of_patches = label_config['num']
                        pickby = label_config.get('pickby', "random")

                    for subtype, subtype_patches in self.output_json[label]['patch'].iteritems():

                        if subtype not in label_subtypes[label]:
                            label_subtypes[label].append(subtype)

                        patches = []
                        for subtype_patch in subtype_patches:
                            patch = dict(subtype_patch)
                            patch['prob'] = float(patch['prob'])
                            patch['label'] = subtype
                            patch['type'] = label
                            patch['orig_path'] = patch['name']
                            patch['name'] = os.path.basename(patch['name'])
                            patch['path'] = label
                            patch['analysis_id'] = str(self.analysis_config['_id'])

                            all_patches[patch['name']] = patch

                            if subtype not in ignore_labels:
                                patches.append(patch)

                        selected_patches = patches

                        if num_of_patches:
                            if pickby == "prob_high":
                                patches = sorted(patches, key=lambda patch: patch['prob'], reverse=True)
                            elif pickby == "prob_low":
                                patches = sorted(patches, key=lambda patch: patch['prob'])
                            selected_patches = patches[:num_of_patches]

                        output_patches_json.extend(selected_patches)

                    name_index_map = {}
                    for i,p in enumerate(output_patches_json):
                        name_index_map[p['name']] = i

                    # Add output patches for this type from other dag procs
                    for patch_name, patch in self.output_json[label].get('output_patch',{}).iteritems():

                        if patch['label'] not in label_subtypes[label]:
                            if patch_name in all_patches:
                                patch['label'] = all_patches[patch_name]['label']
                        patch['type'] = label
                        patch['path'] = label
                        patch['analysis_id'] = str(self.analysis_config['_id'])

                        # Check if the patch is already added in output_patches_json, if yes, just replace it
                        if patch_name in name_index_map:
                                output_patches_json[name_index_map[patch_name]] = patch
                                continue

                        output_patches_json.append(patch)

                output_patches_json_path = os.path.join(self.output_dir, 'output_patches.json')
                outf = DataFile(output_patches_json_path, 'w', self.logger).get_fp()
                json.dump(output_patches_json, outf, ensure_ascii=False)
                outf.close()


            #create report JSON
            report_json_path = None
            if 'all' not in self.report_ignore_fields:
                self.remove_ignore_fields(self.output_json)
                report_json_path = os.path.join(self.output_dir, 'report.json')
                outf = DataFile(report_json_path, 'w', self.logger).get_fp()
                json.dump(self.output_json, outf, ensure_ascii=False)
                outf.close()


            if self.out_queue:
                self.out_queue.put({self.procid: {'status': 'success', 'output_path': self.output_path, 'report_path': report_json_path, 'output_patches_path':output_patches_json_path}})
            self.cleanup()
            self.logger.info('output collation run finished for proc %s' % self.procid)
        except Exception as e:
            self.logger.exception('exception in processing of model collator proc id %s' % self.procid)
            if self.out_queue:
                self.out_queue.put({self.procid: {'status': 'failure'}})

    def cleanup(self):
        pass
