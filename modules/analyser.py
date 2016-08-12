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

class Analyser(multiprocessing.Process):

    def __init__(self, sys_config, module_config, analysis_id, out_queue):
        #initialise multiprocess
        super(Analyser, self).__init__()

        self.sys_config = sys_config
        self.module_config = module_config
        self.analysis_id = analysis_id
        self.name = 'analyser_%s' % self.analysis_id
        self.db_config = self.sys_config.get('db', {})
        self.out_queue = out_queue
        self.start_ts = datetime.utcnow()


    def setup(self):
        self.logger = Logger(self.name, os.path.join(self.sys_config['paths']['log'], '%s_%s.log' % (self.name, datetime.utcnow().strftime('%Y%m%dT%H%M%S'))),'info', False).get_logger()
        #open db connection
        self.dbclient = MDD(self.db_config, self.logger)
        if not self.dbclient:
            raise Exception('could not initialise db connection')
        self.db = self.db_config['database']
        self.analysis_control_table = self.db_config['tables']['analysis']
        self.analyser_table = self.db_config['tables']['analysers']
        self.partner_table = self.db_config['tables']['partners']
        self.extraction_table = self.db_config['tables']['extraction']
        self.analysis_patches_table = self.db_config['tables']['patches']
        self.report_data_table = self.db_config['tables']['report_data']

        #fetch analysis config
        (self.analysis_info, status) = self.dbclient.get_one(self.db, self.analysis_control_table, {'_id': ObjectId(self.analysis_id)})
        if status:
            raise Exception('could not get analysis info from db for id %s' % self.analysis_id)

        #load input config
        self.input_config = self.analysis_info.get('input', {})

        # Set recomputation flag
        self.recomputation = self.analysis_info.get('recompute',False)

        # Past recomputes
        self.recomputes = self.analysis_info.get('recomputes', [])

        # Current pricing plan
        self.pricing_plan = self.analysis_info.get('pricing_plan', {})

        #load extraction id if provided
        self.extraction_id = self.input_config.get('extraction_id', None)
        if self.extraction_id:
            self.extraction_id = ObjectId(self.extraction_id)

        #load extraction query if provided
        self.extraction_query = self.input_config.get('extraction_query', None)

        #load analyser id
        self.analyser_id = self.analysis_info.get('analyser_id', None)
        if not self.analyser_id:
            raise Exception('no analyser provided for analysis id %s' % self.analysis_id)
        #convert to str from ObjectId
        self.analyser_id = str(self.analyser_id)

        #get analyser info from db
        (self.analyser_info, status) = self.dbclient.get_one(self.db, self.analyser_table, {'_id': ObjectId(self.analyser_id)})
        if status:
            raise Exception('could not get analyser info from db for id %s' % self.analyser_id)

        self.analyser_config = None
        try:
            inf = DataFile(os.path.join(self.sys_config['paths']['config'], 'analysers', '%s.json' % self.analyser_info.get('version', '')), 'r', self.logger).get_fp()
            self.analyser_config = json.load(inf)
            inf.close()
        except Exception as e:
            self.logger.error('could not load config for analyser_id %s' % self.analyser_id)
            raise

        #get partner info from db
        (self.partner_info, status) = self.dbclient.get_one(self.db, self.partner_table, {'_id': self.analysis_info.get('partner', None)})
        self.logger.debug("Partner info %s loaded from db " % self.partner_info)
        if status:
            raise Exception('could not get partner info from db for id %s' % self.input_config.get('partner_id', ''))

        #update entry in analysis control table
        (num_updated, status) = self.dbclient.update_data(self.db, self.analysis_control_table, {'_id': ObjectId(self.analysis_id)}, {'$set': {'instance_id': self.sys_config.get('instance_id', ''), 'status': 'launched', 'updt_ts': datetime.utcnow()}})
        if status:
            raise Exception('could not update info for analysis id %s' % self.analysis_id)

        self.analysis_info['analysis_id'] = str(self.analysis_id)
        self.analysis_info['partner'] = self.partner_info
        self.analysis_info['analyser'] = self.analyser_info

        self.logger.info('setup for analysis id %s complete' % self.analysis_id)

    def run(self):
        self.failure_reason = None
        try:
            #setup run
            self.setup()

            #set analysis id in config
            self.input_config['analysis_id'] = self.analysis_id
            analysis_record = {}


            if not self.recomputation:
                if self.extraction_id:
                    #get extraction paths
                    self.logger.info('extraction id provided, skipping registration and extraction')
                    self.get_extraction_from_db()
                elif self.extraction_query:
                    #run extractions with provided query
                    self.logger.info('extraction query provided')
                    ext_config = self.analyser_config.get('extraction', {})
                    ext_config['analysis_id'] = self.analysis_id
                    ext_config['query'] = self.extraction_query
                    self.extraction_info = invoke_data_extraction(self.sys_config, self.module_config, ext_config, self.logger)
                    self.extracted_paths = self.extraction_info['output']
                else:
                    #register data for analysis
                    self.logger.info('registering input data')
                    if 'set_id' not in self.input_config:
                        self.input_config['set_id'] = self.analysis_info.get('sample_id', None)
                    self.input_config['partner_id'] = self.analysis_info.get('partner', {}).get('_id', None)
                    self.input_config['subm_by'] = self.analysis_info.get('subm_by', None)
                    dr = DataRegistrator(self.sys_config, self.module_config, self.input_config, self.logger)
                    dr.register_data()
                    (num_updated, status) = self.dbclient.update_data(self.db, self.analysis_control_table, {'_id': ObjectId(self.analysis_id)}, {'$set': {'status': 'registration complete', 'updt_ts': datetime.utcnow()}})
                    if status:
                        self.failure_reason = "Image registration failed"
                        raise Exception('could not update status in db for analysis_id %s' % self.analysis_id)

                    #run required extractions
                    self.logger.info('extracting required patches')
                    ext_config = self.analyser_config.get('extraction', {})
                    ext_config['analysis_id'] = self.analysis_id
                    ext_config['query'] = {'analysis_id': self.analysis_id}
                    self.extraction_info = invoke_data_extraction(self.sys_config, self.module_config, ext_config, self.logger)
                    self.extracted_paths = self.extraction_info['output']

                analysis_record['extr_id'] = self.extraction_info['_id']

            else:
                # Get analysis directory
                DataFile(os.path.join(self.sys_config['paths']['analysis'],str(self.analysis_id)), 'r', self.logger, folder=True)
                if not os.path.exists(os.path.join(self.sys_config['paths']['analysis'],str(self.analysis_id),"recomputation")):
                    os.makedirs(os.path.join(self.sys_config['paths']['analysis'],str(self.analysis_id),"recomputation"))
                updated_labels_cursor,status = self.dbclient.get_data(self.db,self.analysis_patches_table,{'analysis_id':self.analysis_id, '$or': [{'new_label': {'$exists': True}},{ 'modified_label': {'$exists':True}}]})
                self.updated_labels = {}
                for updated_label in updated_labels_cursor:
                    self.updated_labels[updated_label['name']] = updated_label

                self.extraction_id = self.analysis_info.get('extr_id', None)
                self.get_extraction_from_db()

            #update status
            analysis_record['status'] = 'extraction complete'
            analysis_record['updt_ts'] = datetime.utcnow()
            if "analysis_output" in self.sys_config['sync_entities']:
                analysis_record['sync_status'] = 'sync_pending:' + self.sys_config.get("instance_id", "")
            else:
                analysis_record['sync_status'] = 'not_synced'
            (num_updated, status) = self.dbclient.update_data(self.db, self.analysis_control_table, {'_id': ObjectId(self.analysis_id)}, {'$set': analysis_record})
            if status:
                self.failure_reason = "Cell extraction failed"
                raise Exception('could not update status in db for analysis id %s' % self.analysis_id)

            #setup for DAG processing
            self.analyser_config['inputs'] = self.extracted_paths
            if self.recomputation:
                self.analyser_config['output'] = os.path.join(self.sys_config['paths']['analysis'], str(self.analysis_id), 'recomputation')
                self.analyser_config['orig_output'] = os.path.join(self.sys_config['paths']['analysis'], str(self.analysis_id))
                self.analysis_info['updated_labels'] = self.updated_labels
            else:
                self.analyser_config['output'] = os.path.join(self.sys_config['paths']['analysis'], str(self.analysis_id))
            self.analysis_info['num_input'] = len(self.extraction_info.get('file_info', []))
    
            #process analyser DAG to create various outputs
            self.logger.info('executing configured processes on extracted patches')
            self.dag = DAGExecutor(self.sys_config, self.analyser_config, self.analysis_info, self.logger,recomputation=self.recomputation)
            (self.dag_output_path, self.report_data_path, self.output_patches_path) = self.dag.execute()
            self.cleanup()
            self.logger.info('all processing executed successfully')
            #load report data
            self.report_data = {}
            if self.report_data_path:
                inf = DataFile(self.report_data_path, 'r', self.logger).get_fp()
                self.report_data = json.load(inf)
                inf.close()


            #load output patches data
            self.output_patches = {}
            if not self.recomputation and self.output_patches_path:
                inf = DataFile(self.output_patches_path, 'r', self.logger).get_fp()
                self.output_patches = json.load(inf)
                inf.close()
                self.extracted_images_path = os.path.commonprefix(filter(None,map(lambda x: os.path.dirname(x['images']),self.extracted_paths.values())))
                self.analysis_patches_path = os.path.join(self.analyser_config['output'], "analysis_patches")


            for output_patch in self.output_patches:
                # Copy the file to analysis folder
                if not os.path.exists(os.path.join(self.analysis_patches_path, output_patch.get('path',''))):
                    os.makedirs(os.path.join(self.analysis_patches_path, output_patch.get('path','')))
                shutil.copyfile(output_patch.get('orig_path', ''), os.path.join(self.analysis_patches_path, output_patch.get('path',''), output_patch.get('name','')))

                # Write output patch info to db
                output_patch['analysis_id'] = ObjectId(output_patch['analysis_id'])
                (id, status) = self.dbclient.post_data(self.db, self.analysis_patches_table, output_patch)
                if status:
                    self.failure_reason = "Error writing patches to database"
                    raise Exception('could not write patches to  in db for analysis id %s' % str(self.analysis_id))

            # Write report data to db
            record = {'analysis': self.analysis_id, 'report_data': self.report_data, 'modified':self.recomputation, 'updt_ts': datetime.utcnow()}
            (num_updated, status) = self.dbclient.update_data(self.db, self.report_data_table,{'analysis':self.analysis_id, 'modified':self.recomputation}, {'$set':record}, upsert=True)
            if status:
                self.failure_reason = "Error updating report data in the database"
                raise Exception('could not update report data in db for analysis id %s' % str(self.analysis_id))

            if self.recomputation:
                for updated_label in self.updated_labels.values():
                    updated_label['modified_label'] = updated_label.get('new_label',updated_label.get('modified_label',None))
                    updated_label['modified_by'] = updated_label.get('updt_by', updated_label.get('modified_by',None))
                    if "new_label" in updated_label:
                        del updated_label['new_label']
                    if "updt_by" in updated_label:
                        del updated_label['updt_by']
                    (num_updated, status) = self.dbclient.update_data(self.db, self.analysis_patches_table, {'_id': ObjectId(updated_label['_id'])}, {'$set':updated_label, '$unset':{'new_label':'', 'updt_by':''}})
                    if status:
                        self.failure_reason = "Error updating the database"
                        raise Exception('could not update recomputed patches in db for analysis id %s' % str(self.analysis_id))

            record = {'status': 'success', 'updt_ts': datetime.utcnow(), 'output': self.dag_output_path, 'modified':self.recomputation, 'failure_reason':''}
            if not self.recomputation and self.output_patches_path:
                record['patches_path'] = self.analysis_patches_path
            if self.recomputation:
                self.recomputes.append({'ts': datetime.utcnow(), 'pricing': self.pricing_plan.get('_id',''), 'unit_price': self.pricing_plan.get('add_unit_price',{}).get('recomputes',{}).get('unit_price','')})
                record['recomputes'] = self.recomputes
                record['size'] = self.analysis_info['size']

            else:
                record['orig_pricing_plan'] = {}
                record['orig_pricing_plan']['_id'] = self.pricing_plan.get('_id', '')
                record['orig_pricing_plan']['unit_price'] = self.pricing_plan.get('unit_price', '')
                record['size'] = {}
                record['size']['extraction'] = self.extraction_info.get('extraction_size',0)
                record['size']['input'] = self.extraction_info.get('input_size',0)
                record['num_files'] = self.extraction_info.get("no_of_files",0)

            record['size']['analysis'] = DataFile(os.path.join(self.sys_config['paths']['analysis'],str(self.analysis_id)), 'r', self.logger, folder=True).get_size()

            if 'analysis_output' in self.sys_config.get('sync_entities', []):
                record['status'] = 'data sync in progress'


            record['analysis_time'] = (datetime.utcnow() - self.start_ts).total_seconds()

            (num_updated, status) = self.dbclient.update_data(self.db, self.analysis_control_table, {'_id': ObjectId(self.analysis_id)}, {'$set': record})
            if status:
                self.failure_reason = "Error updating the database"
                raise Exception('could not update status in db for analysis id %s' % str(self.analysis_id))

            if "analysis_output" in self.sys_config['sync_entities']:
                self.schedule_for_sync()

            if self.out_queue:
                self.out_queue.put({str(self.analysis_id): {'status': 'success'}})

        except Exception as e:
            #update failure status is db
            if self.failure_reason == None:
                self.failure_reason = "Analysis failed"
            if self.recomputation:
                status = "recompute failure"
            else:
                status = "failure"
            self.logger.exception('caught exception during analysis run of id %s' % str(self.analysis_id))
            (num_updated, status) = self.dbclient.update_data(self.db, self.analysis_control_table, {'_id': ObjectId(self.analysis_id)}, {'$set': {'status': status, 'failure_reason': self.failure_reason, 'updt_ts': datetime.utcnow()}})
            if status:
                self.logger.error('could not update failure status in db for analysis id %s' % str(self.analysis_id))
            if self.out_queue:
                self.out_queue.put({str(self.analysis_id): {'status': 'failure'}})

    def schedule_for_sync(self):
        self.logger.debug("Scheduling the analysis for sync. Invoked function DataExtraction.schedule_for_sync")
        database = self.sys_config["db"]["database"]
        collection = self.sys_config["db"]["tables"]["sync"]
        analysis_path = os.path.join(self.sys_config["paths"]["analysis"], str(self.analysis_id))

        sync_record = {}
        sync_record['instance_id'] = self.sys_config.get('instance_id', '')
        sync_record['ts'] = datetime.utcnow()
        sync_record['status'] = 'pending'
        sync_record['sync_type'] = 'analysis'
        sync_record['source_table'] = 'analysis_control'
        sync_record['source_key'] = self.analysis_id
        sync_record['source_status_key'] = 'status'
        sync_record['source_status_value'] = 'success'
        sync_record['sync_paths'] = []
        sync_record['sync_paths'].append(analysis_path)
        (id, status) = self.dbclient.post_data(database, collection, sync_record)

        if status:
            raise Exception("Failed to schedule extraction for sync")

    def cleanup(self):
        self.dag.cleanup()

    def get_extraction_from_db(self):
        (self.extraction_info, status) = self.dbclient.get_one(self.db, self.extraction_table, {'_id': self.extraction_id})
        if status:
            self.failure_reason = "Extraction id provided in input doesn't exist"
            raise Exception('could not fetch info from db for extraction id: %s' % str(self.extraction_id))
        if self.extraction_info.get('instance_id', '') != self.sys_config.get('instance_id', ''):
            self.failure_reason = "Extraction info not synced"
            raise Exception('instance id of extraction (%s) does not match local instance id (%s)' % (self.extraction_info.get('instance_id', ''), self.sys_config.get('instance_id', '')))
        self.extracted_paths = self.extraction_info['output']
        self.logger.info('extraction paths fetched from db')


