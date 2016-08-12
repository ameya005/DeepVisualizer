#!/usr/bin/env python

import os,sys
import codecs
import json
from basedagproc import BaseDAGProc
from modules.utils.fsutils import DataFile


class ModelCollatorDAGProc(BaseDAGProc):

    def __init__(self, sys_config, config, analysis_config, out_queue, proc_configs, logger=None):
        #invoke base constructor
        super(ModelCollatorDAGProc, self).__init__(sys_config, config, analysis_config, out_queue, proc_configs, logger=logger)

        #validate_config
        self.output_dir = self.config.get('output', None)
        if not self.output_dir:
            raise Exception('no output path provided in config for DAG proc id %s' % self.procid)
        self.output_dir = os.path.join(self.output_dir, self.procid)
        self.output_path = None
        self.output_label = self.config.get('output_label', '')
        if not self.output_label:
            raise Exception('no output label provided for model collator proc id %s' % self.procid)

        self.unclassified_prob_cutoff = self.config.get('unclassified_prob_cutoff', {})
        self.unclassified_label = self.config.get('unclassified_label', '')

        self.label_whitelist = self.config.get('labels_from', {})
        if not self.label_whitelist:
            raise Exception('no label whitelist provided for model collator proc id %s' % self.procid)

        self.schema = {}
        cnt = 0
        for field_name in self.config.get('schema', []):
            self.schema[field_name] = cnt
            cnt += 1
        for name in ['name', 'label', 'prob']:
            if name not in self.schema:
                raise Exception('provided schema in config does not contain required field %s for proc id %s' % (name, self.procid))


        # For patch label changes
        self.output_labels = self.config.get('output_labels',[])
        self.updated_labels = self.analysis_config.get('updated_labels', {})

        self.total_ignore_labels = set(self.config.get('total_ignore_labels', []))

        self.sep = self.config.get('sep', ',')

        self.confusion = self.config.get('confusion', False)
        self.confusion_threshold = 1.0
        if self.confusion:
            self.confusion_threshold = self.config.get('confusion_threshold', 0.3)


        self.total_count_factor = self.config.get('total_count_factor', -1.0)
        self.total_count_pct_factor = self.config.get('total_count_pct_factor', True)

        self.input_attrs = self.config.get('input_attributes', False)
        self.input_attrs_keys = self.config.get('input_attributes_key', '').split(':')

        self.pct_unit = self.config.get('pct_unit', None)

        #dedupe labels
        self.dedupe_labels_preference = self.config.get('dedupe_labels_preference', [])
        self.dedupe_offset = self.config.get('dedupe_offset', 2)

        self.logger.info('model collator DAG proc id %s initialised successfully' % self.procid)

    def run(self):
        try:
            self.output_info_base = {'count': {}, 'pct': {}, 'patch': {}}
            self.output_info = {self.output_label: self.output_info_base}
            total_counts_fovs = None
            if self.total_count_factor > 0:
                total_counts_fovs = {}
                self.output_info_base['tc'] = {}
            if self.confusion:
                self.output_info_base['confusion'] = {'count': {}, 'patch': {}}
            if self.input_attrs:
                self.input_attrs_base = self.output_info
                for k in self.input_attrs_keys:
                    if not k:
                        continue
                    self.input_attrs_base = self.input_attrs_base.setdefault(k, {})
            else:
                self.input_attrs_base = None
    
            #get outputs of dependencies and load
            total_count = 0
            for pid in self.label_whitelist:
                labels = set(self.label_whitelist.get(pid, []))
                if pid not in self.proc_configs:
                    raise Exception('dependency proc id %d not found in proc_configs list')
                fpath = self.proc_configs[pid].get('output_path', None)
                if not fpath or not os.path.exists(fpath):
                    raise Exception('output path %s of proc id %s not found' % (fpath, pid))
                inf = DataFile(fpath, 'r', self.logger).get_fp()
                for line in inf:
                    fields = line.strip().split(self.sep)
                    label = fields[self.schema['label']]
                    name = fields[self.schema['name']]
                    prob = float(fields[self.schema['prob']])
                    allprobs = {}
                    second_label = None
                    second_prob = 0.0
                    if self.confusion:
                        for vals in fields[self.schema['allprobs']].strip().split('|'):
                            (l,p) = vals.split(':', 1)
                            allprobs[l] = float(p)
                        for (l,p) in allprobs.iteritems():
                            if p == prob:
                                continue
                            if p > second_prob:
                                second_prob = p
                                second_label = l
    
                    if label in labels or 'all' in labels:
                        orig_label = ''
                        if prob < self.unclassified_prob_cutoff.get(label, self.unclassified_prob_cutoff.get('default', 0.0)):
                            orig_label = label
                            label = self.unclassified_label
                        info = {'name':name, 'prob':str(prob)}
                        if orig_label:
                            info['orig_label'] = orig_label
                        if None is not self.pct_unit:
                            self.output_info_base['count'].setdefault(label, {})['val'] = self.output_info_base['count'].get(label, {}).get('val', 0)+1
                        else:
                            self.output_info_base['count'][label] = self.output_info_base['count'].get(label, 0)+1
                        if label not in self.total_ignore_labels:
                            total_count += 1
                        #calculate total counts of fovs
                        fov_name = os.path.basename(name).split('_')[0]
                        if self.total_count_factor > 0 and (label not in self.total_ignore_labels or label == self.unclassified_label):
                            total_counts_fovs[fov_name] = total_counts_fovs.get(fov_name, 0)+1
                        #add input attributes
                        if None is not self.input_attrs_base and (label not in self.total_ignore_labels or label == self.unclassified_label):
                            input_info = self.input_attrs_base.setdefault(fov_name, {})
                            input_info[self.output_label] = input_info.get(self.output_label, 0)+1
                        self.output_info_base['patch'].setdefault(label, {})[os.path.basename(info['name'])] = info
                        if not orig_label and self.confusion and second_prob >= self.confusion_threshold:
                            #add to confusion classes
                            info = {'name':name}
                            confusion_key = '-'.join(sorted([label, second_label]))
                            self.output_info_base['confusion']['count'][confusion_key] = self.output_info_base['confusion']['count'].get(confusion_key, 0)+1
                            info[label] = str(prob)
                            info[second_label] = str(second_prob)
                            self.output_info_base['confusion']['patch'].setdefault(confusion_key, []).append(info)
                inf.close()


            #dedupe
            if self.dedupe_labels_preference:
                dedupe_set = set([])
                for i in xrange(len(self.dedupe_labels_preference)):
                    label = self.dedupe_labels_preference[i]
                    if label not in self.output_info_base['patch']:
                        continue
                    label_new_cells = {}
                    for key,cell in self.output_info_base['patch'].get(label,{}).iteritems():
                        name_toks = os.path.splitext(os.path.basename(cell['name']))[0].split('_')
                        dupkey = '%s_%s_%s' % (name_toks[0], name_toks[2], name_toks[3])
                        if dupkey in dedupe_set:
                            self.logger.info('removing %s in duplicate check' % (os.path.basename(cell['name'])))
                            #remove this cell and its counts
                            if None is not self.pct_unit:
                                self.output_info_base['count'][label]['val'] = self.output_info_base['count'][label]['val']-1
                            else:
                                self.output_info_base['count'][label] = self.output_info_base['count'][label]-1
                            #adjust total count
                            if label not in self.total_ignore_labels:
                                total_count -= 1
                            #adjust total counts of fovs
                            fov_name = os.path.basename(cell['name']).split('_')[0]
                            if self.total_count_factor > 0 and (label not in self.total_ignore_labels or label == self.unclassified_label):
                                total_counts_fovs[fov_name] = total_counts_fovs[fov_name]-1
                            #adjust input attributes
                            if None is not self.input_attrs_base and (label not in self.total_ignore_labels or label == self.unclassified_label):
                                input_info = self.input_attrs_base.setdefault(fov_name, {})
                                input_info[self.output_label] = input_info[self.output_label]-1
                        else:
                            label_new_cells[key] = cell
                            #if not last label in preference order, add to hash
                            if i != len(self.dedupe_labels_preference)-1:
                                cx,cy = int(name_toks[2]), int(name_toks[3])
                                dedupe_set.add(dupkey)
                                for p in xrange(self.dedupe_offset+1):
                                    for tx in [cx+p,cx-p]:
                                        for q in xrange(self.dedupe_offset+1):
                                            for ty in [cy+q,cy-q]:
                                                dkey = '%s_%d_%d' % (name_toks[0], tx, ty)
                                                dedupe_set.add(dkey)
                    #update new label list
                    self.output_info_base['patch'][label] = label_new_cells

            # Apply label updates
            for patch_name, patch_info in self.updated_labels.iteritems():
                # Remove patches with original label in this model collator's output labels
                label = patch_info['label']
                new_label = patch_info.get('new_label', patch_info.get('modified_label',None))
                prob = patch_info['prob']
                orig_path =  patch_info['orig_path']
                if patch_name in self.output_info_base['patch'].get(patch_info['label'], {}):
                    # Update counts
                    if None is not self.pct_unit:
                        self.output_info_base['count'][label]['val'] = self.output_info_base['count'][label]['val']-1
                    else:
                        self.output_info_base['count'][label] = self.output_info_base['count'][label]-1
                    # adjust total count
                    if label not in self.total_ignore_labels:
                        total_count -= 1
                    # Adjust input counts
                    fov_name = patch_name.split('_')[0]
                    if self.total_count_factor > 0 and (label not in self.total_ignore_labels or label == self.unclassified_label):
                        total_counts_fovs[fov_name] = total_counts_fovs[fov_name] - 1
                    if None is not self.input_attrs_base and (label not in self.total_ignore_labels or label == self.unclassified_label):
                        input_info = self.input_attrs_base.setdefault(fov_name, {})
                        input_info[self.output_label] = input_info[self.output_label] - 1
                    del self.output_info_base['patch'][label][patch_name]

                # Add patches with new label in this model collator's output labels
                if new_label in self.output_labels or (new_label == self.unclassified_label and label in self.output_labels):
                    self.output_info_base['patch'].setdefault(new_label, {})[patch_name] = {'name': orig_path, 'prob':prob, 'orig_label':label, 'corrected':True,'label':new_label}
                    if None is not self.pct_unit:
                        self.output_info_base['count'].setdefault(new_label,{})['val'] = self.output_info_base['count'].get(new_label, {}).get('val',0) + 1
                    else:
                        self.output_info_base['count'][new_label] = self.output_info_base['count'].get(new_label, 0) + 1

                    # adjust total count
                    if new_label not in self.total_ignore_labels:
                        total_count += 1

                    fov_name = patch_name.split('_')[0]
                    if self.total_count_factor > 0 and (new_label not in self.total_ignore_labels or new_label == self.unclassified_label):
                        total_counts_fovs[fov_name] = total_counts_fovs.get(fov_name, 0) + 1
                    if None is not self.input_attrs_base and (new_label not in self.total_ignore_labels or new_label == self.unclassified_label):
                        input_info = self.input_attrs_base.setdefault(fov_name, {})
                        input_info[self.output_label] = input_info.get(self.output_label,0) + 1

            for label, patch_dict in self.output_info_base['patch'].iteritems():
                self.output_info_base['patch'][label] = patch_dict.values()

            #fill up pct values
            total_count_per_unit_vol = 0.0
            if self.total_count_factor > 0:
                fov_sum = 0
                fov_cnt = 0
                fov_avg = 0.0
                for k,v in total_counts_fovs.iteritems():
                    fov_sum += v
                    fov_cnt += 1
                if fov_cnt > 0:
                    fov_avg = (1.0*fov_sum)/fov_cnt
                    total_count_per_unit_vol = self.total_count_factor*fov_avg
                self.logger.info('total counts fovs: %d, avg: %f, tc per unit vol: %f' % (fov_cnt, fov_avg, total_count_per_unit_vol))
            if None is not self.pct_unit:
                self.output_info_base['total'] = {'val': total_count}
            else:
                self.output_info_base['total'] = total_count
            for k,v in self.output_info_base['count'].iteritems():
                if k not in self.total_ignore_labels:
                    if None is not self.pct_unit:
                        self.output_info_base['pct'][k] = {'val': round((v['val']*100.)/total_count, 2), 'unit': self.pct_unit}
                    else:
                        self.output_info_base['pct'][k] = round((v*100.)/total_count, 2)
                    if self.total_count_factor > 0:
                        if self.total_count_pct_factor:
                            if None is not self.pct_unit:
                                self.output_info_base['tc'][k] = int(((v['val']*1.)/total_count)*total_count_per_unit_vol)
                            else:
                                self.output_info_base['tc'][k] = int(((v*1.)/total_count)*total_count_per_unit_vol)
                        else:
                            self.output_info_base['tc'][k] = int(total_count_per_unit_vol)
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            self.output_path = os.path.join(self.output_dir, 'model_collated.json')
            outf = DataFile(self.output_path, 'w', self.logger).get_fp()
            json.dump(self.output_info, outf, ensure_ascii=False)
            outf.close()
            if self.out_queue:
                self.out_queue.put({self.procid: {'status': 'success', 'output_path': self.output_path}})
            self.cleanup()
            self.logger.info('model collation run finished for proc_id %s' % self.procid)
        except Exception as e:
            self.logger.exception('exception in processing of model collator proc id %s' % self.procid)
            if self.out_queue:
                self.out_queue.put({self.procid: {'status': 'failure'}})

    def cleanup(self):
        pass
