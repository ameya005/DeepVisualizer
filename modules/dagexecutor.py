#!/usr/bin/env python

import os,sys
import json
import multiprocessing
from dagprocs import basedagproc
import shutil
import codecs

class DAGExecutor(object):

    def __init__(self, sys_config, analyser_config, analysis_config, logger, recomputation=False):
        self.logger = logger
        self.sys_config = sys_config
        self.analyser_config = analyser_config
        self.analysis_config = analysis_config
        self.out_queue = multiprocessing.Queue(len(self.analyser_config.get('processes', [])))
        self.max_running = self.sys_config.get('max_parallel_per_dag', 4)
        self.recomputation = recomputation

        #load all processes
        self.proc_configs = {}
        self.done_procs = {}
        self.running_procs = {}
        for proc in self.analyser_config.get('processes', []):
            #set input and output paths if not present
            if 'input' not in proc:
                if 'input_types' in proc:
                    proc['input'] = []
                    for input_info in proc['input_types']:
                        (input_type,input_format) = input_info.split(':')
                        proc['input'].append(self.analyser_config['inputs'][input_type][input_format])
            if 'output' not in proc:
                proc['output'] = self.analyser_config['output']
            self.proc_configs[proc.get('id', None)] = proc
        self.final_output_path = None
        self.final_report_data_path = None
        self.final_output_patches_path = None

        if self.recomputation:
            self.recompute_exclude_procs = self.analyser_config.get('recompute_exclude',[])
            self.label_updates = self.analyser_config.get('label_update',[])

        self.logger.info('initialisation complete')

    def killall(self):
        for (pid,proc) in self.running_procs.iteritems():
            if proc.is_alive():
                self.logger.info('killing proc %s' % pid)
                proc.terminate()

    def can_run(self, procid, proc_config):
        for pid in proc_config.get('depends', []):
            if pid not in self.done_procs:
                self.logger.info('proc %s cannot run because dependency %s has not finished yet' % (procid, pid))
                return False
        self.logger.info('proc %s can run, no outstanding dependency left' % (procid))
        return True

    def is_remaining(self):
        if len(self.proc_configs)  == len(self.done_procs):
            return False
        return True

    def execute(self):
        while self.is_remaining():
            #check if more can be run
            for (pid, proc_config) in self.proc_configs.iteritems():
                if len(self.running_procs) >= self.max_running:
                    break
                if pid in self.done_procs or pid in self.running_procs:
                    continue
                if not self.can_run(pid, proc_config):
                    continue
                if self.recomputation and pid in self.recompute_exclude_procs:
                    # Process shouldn't be run, copy already existing process output directory to recomputation directory
                    self.logger.info("skipping DAG proc id %s " % pid)
                    # Recomputation has not run before, copy process output directory from original analysis run
                    if not os.path.exists(os.path.join(proc_config['output'], pid)):
                        shutil.copytree(os.path.join(self.analyser_config['orig_output'],pid), os.path.join(proc_config['output'], pid))

                    proc = basedagproc.factory(proc_config.get('type', ''), self.sys_config, proc_config, self.analysis_config, self.out_queue, self.proc_configs)
                    recompute_output_path = proc.recompute()
                    proc_config['output_path'] = recompute_output_path
                    self.done_procs[pid] = proc
                    continue
                self.logger.info('starting DAG proc id %s' % pid)
                proc = basedagproc.factory(proc_config.get('type', ''), self.sys_config, proc_config, self.analysis_config, self.out_queue, self.proc_configs)
                proc.start()
                self.running_procs[pid] = proc
            #wait for a proc to send a msg and finish
            out_info = self.out_queue.get()
            for pid in out_info:
                proc = self.running_procs.get(pid, None)
                if out_info[pid].get('status', 'failure') != 'success':
                    self.killall()
                    raise Exception('DAG process id %s failed' % pid)
                else:
                    self.logger.info('DAG process id %s completed successfully' % pid)
                #join pid
                proc.join()
                #transfer
                for key,val in out_info[pid].iteritems():
                    self.proc_configs[pid][key] = val
                if self.proc_configs[pid]['type'] == 'output':
                    self.final_output_path = self.proc_configs[pid]['output_path']
                    self.final_report_data_path = self.proc_configs[pid].get('report_path', None)
                    self.final_output_patches_path = self.proc_configs[pid].get('output_patches_path', None)
                self.done_procs[pid] = self.running_procs.pop(pid)
        self.logger.info('finished executing all procs')
        return (self.final_output_path, self.final_report_data_path, self.final_output_patches_path)

    def cleanup(self):
        pass
