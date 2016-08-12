#!/usr/bin/env python

import os,sys
import json
import time
import glob
from datetime import datetime
from bson.objectid import ObjectId
from mdd import MDD
from dataregistration import DataRegistrator
from dataextraction import invoke_data_extraction
from dagexecutor import DAGExecutor
from modules.utils.fsutils import DataFile
from modules.utils.logger import Logger
import multiprocessing
import shutil

class Ingestor(multiprocessing.Process):

    def __init__(self, sys_config, module_config, ingestion_id, out_queue):
        #initialise multiprocess
        super(Ingestor, self).__init__()

        self.sys_config = sys_config
        self.module_config = module_config
        self.ingestion_id = ingestion_id
        self.name = 'ingestor_%s' % self.ingestion_id
        self.db_config = self.sys_config.get('db', {})
        self.out_queue = out_queue

    def setup(self):
        self.logger = Logger(self.name, os.path.join(self.sys_config['paths']['log'], '%s_%s.log' % (self.name, datetime.utcnow().strftime('%Y%m%dT%H%M%S'))),'info', False).get_logger()
        #open db connection
        self.dbclient = MDD(self.db_config, self.logger)
        if not self.dbclient:
            raise Exception('could not initialise db connection')
        self.db = self.db_config['database']
        self.input_control_table = self.db_config['tables']['input']
        self.ingestion_control_table = self.db_config['tables']['ingestion']

        #fetch ingestion config
        (self.ingestion_info, status) = self.dbclient.get_one(self.db, self.ingestion_control_table, {'_id': ObjectId(self.ingestion_id)})
        if status:
            raise Exception('could not get ingestion info from db for id %s' % self.ingestion_id)

        #update entry in ingestion control table
        (num_updated, status) = self.dbclient.update_data(self.db, self.ingestion_control_table, {'_id': ObjectId(self.ingestion_id)}, {'$set': {'instance_id': self.sys_config.get('instance_id', ''), 'status': 'processing', 'updt_ts': datetime.utcnow()}})
        if status:
            raise Exception('could not update info for ingestion id %s' % self.ingestion_id)

        self.ingestion_info['ingestion_id'] = str(self.ingestion_id)
        self.logger.info('setup for ingestion id %s complete' % self.ingestion_id)

    def run(self):
        self.failure_reason = None
        try:
            #setup run
            self.setup()

            #register data for ingestion
            self.logger.info('registering input data')
            self.input_config = self.ingestion_info.get('input', {})
            if 'set_id' not in self.input_config:
                self.input_config['set_id'] = self.ingestion_info.get('sample_id', self.ingestion_info.get('case_id', None))
                self.input_config['desc'] = self.ingestion_info.get('desc', self.ingestion_info.get('sample_desc', self.ingestion_info.get('case_desc', None)))
            self.input_config['partner_id'] = self.ingestion_info.get('partner', None)
            self.input_config['subm_by'] = self.ingestion_info.get('subm_by', None)
            dr = DataRegistrator(self.sys_config, self.module_config, self.input_config, self.logger)
            dr.register_data()
            (num_updated, status) = self.dbclient.update_data(self.db, self.ingestion_control_table, {'_id': ObjectId(self.ingestion_id)}, {'$set': {'status': 'success', 'updt_ts': datetime.utcnow()}})
            if status:
                self.failure_reason = "ingestion failed"
                raise Exception('could not update status in db for ingestion_id %s' % self.ingestion_id)
            if self.out_queue:
                self.out_queue.put({str(self.ingestion_id): {'status': 'success'}})

        except Exception as e:
            #update failure status in db
            if self.failure_reason == None:
                self.failure_reason = "ingestion failed"
            status = "failure"
            self.logger.exception('caught exception during ingestion run of id %s' % str(self.ingestion_id))
            (num_updated, status) = self.dbclient.update_data(self.db, self.ingestion_control_table, {'_id': ObjectId(self.ingestion_id)}, {'$set': {'status': status, 'failure_reason': self.failure_reason, 'updt_ts': datetime.utcnow()}})
            if status:
                self.logger.error('could not update failure status in db for ingestion id %s' % str(self.ingestion_id))
            if self.out_queue:
                self.out_queue.put({str(self.ingestion_id): {'status': 'failure'}})

    def cleanup(self):
        pass
