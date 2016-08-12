#!/usr/bin/env python

import os,sys
import numpy
from modules.mdd import MDD
from modules.inputreaders import baseinputreader

class ImageParamUtils(object):

    def __init__(self, sys_config, logger):
        self.sys_config = sys_config
        self.logger = logger
        self.db_config = self.sys_config.get('db', {})
        self.dbclient = MDD(self.db_config, self.logger)
        if not self.dbclient:
            raise Exception('could not open db connection')
        self.db = self.db_config['database']
        self.input_control_table = self.db_config['tables']['input']

    def get_input_file_lcn(self, input_id, input_set_id):
        #get all valid input files
        query = {'file_name': input_id, 'set_id': input_set_id, 'status': {'$in': ['success', 'sync_pending:%s' % self.sys_config.get('instance_id', '')]}}
        input_files,status = self.dbclient.get_data(self.db, self.input_control_table, query)
        if status or None is input_files:
            raise Exception('could not fetch all files in set %s from DB' % (input_set_id))
        total_means = None
        total_stds = None
        total_count = 0
        self.logger.info('calculating average lcn for %d file(s) of input id %s' % (input_files.count(), input_id))
        for f in input_files:
            input_path = os.path.join(f['file_dest'], '%s%s' % (f['file_name'], f['file_ext']))
            input_reader = baseinputreader.factory(f['file_type'], self.logger, input_path, in_ppm=[f['ppm_x'],f['ppm_y']], out_ppm=[f['ppm_x'],f['ppm_y']])
            (means,stds) = input_reader.get_lcn_params()
            total_means = means if None is total_means else total_means + means
            total_stds = stds if None is total_stds else total_stds + stds
            total_count += 1
            input_reader.close()

        if total_count == 0:
            return (numpy.asarray([]), numpy.asarray([]))

        return (total_means/total_count, total_stds/total_count)

    def get_input_file_histogram(self, input_id, input_set_id):
        #get all valid input files
        query = {'file_name': input_id, 'set_id': input_set_id, 'status': {'$in': ['success', 'sync_pending:%s' % self.sys_config.get('instance_id', '')]}}
        input_files,status = self.dbclient.get_data(self.db, self.input_control_table, query)
        if status or None is input_files:
            raise Exception('could not fetch all files in set %s from DB' % (input_set_id))
        total_hist = None
        total_count = 0
        self.logger.info('calculating average histogram for %d file(s) of input id %s' % (input_files.count(), input_id))
        for f in input_files:
            input_path = os.path.join(f['file_dest'], '%s%s' % (f['file_name'], f['file_ext']))
            input_reader = baseinputreader.factory(f['file_type'], self.logger, input_path, in_ppm=[f['ppm_x'],f['ppm_y']], out_ppm=[f['ppm_x'],f['ppm_y']])
            hist = input_reader.get_cumulative_histogram()
            if None is total_hist:
                total_hist = hist
            else:
                total_hist += hist
            total_count += 1
            input_reader.close()

        if total_count > 0:
            return (total_hist/total_count).ravel()
        else:
            return []

    def get_input_set_histogram(self, input_set_id):
        #get all valid input files
        query = {'set_id': input_set_id, 'status': {'$in': ['success', 'sync_pending:%s' % self.sys_config.get('instance_id', '')]}}
        input_files,status = self.dbclient.get_data(self.db, self.input_control_table, query)
        if status or None is input_files:
            raise Exception('could not fetch all files in set %s from DB' % (input_set_id))
        total_hist = None
        total_count = 0
        self.logger.info('calculating average histogram for %d file(s) in input set %s' % (input_files.count(), input_set_id))
        for f in input_files:
            input_path = os.path.join(f['file_dest'], '%s%s' % (f['file_name'], f['file_ext']))
            input_reader = baseinputreader.factory(f['file_type'], self.logger, input_path, in_ppm=[f['ppm_x'],f['ppm_y']], out_ppm=[f['ppm_x'],f['ppm_y']])
            hist = input_reader.get_cumulative_histogram()
            if None is total_hist:
                total_hist = hist
            else:
                total_hist += hist
            total_count += 1
            input_reader.close()

        if total_count > 0:
            return (total_hist/total_count).ravel()
        else:
            return []

    def get_histogram_mapping(self, ref_hist, input_hist):
        pixmap = numpy.zeros((256,), dtype=numpy.uint8)
        for i in xrange(256):
            if i <= 10:
                pixmap[i] = i
                continue
            pval = input_hist[i]
            for j in xrange(256):
                if ref_hist[j] >= pval:
                    break
            pixmap[i] = j
        return pixmap
