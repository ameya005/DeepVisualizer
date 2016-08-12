#!/usr/bin/env python
# coding=utf-8
''' dataregistration.py: to register a new input file in KŪRMA'''

__author__ = "Rohit K. Pandey"
__copyright__ = "Copyright 2015, SigTuple Technologies Pvt. Ltd"
__version__ = "0.1"
__email__ = "rohit@sigtuple.com"

'''
This module will register a new data file etc.
into the platform. This will use metadata information
provided along with the data file and that present
as an attribute in the datafile itself. E.g. - slide/image
properties etc.'''


#from hurry.filesize import size
import json
import datetime
import os
from modules.mdd import MDD
from bson import json_util
from bson import ObjectId
import shutil
import sys
import boto3
from modules.inputreaders import baseinputreader
from modules.utils.fsutils import DataFile

class DataRegistrator(object):

    def __init__(self, sys_cfg, mod_cfg, user_cfg, logger):
        self.logger = logger
        self.sys_cfg = sys_cfg
        self.user_cfg = user_cfg
        self.mod_cfg = mod_cfg
        self.mdd = MDD(self.sys_cfg["db"],self.logger)
        if not self.mdd:
            raise Exception('could not initialise db connection')
        #validate input files
        for finfo in self.user_cfg.get('file_list', []):
            #fetch from remote data store if not available locally
            infp = DataFile(finfo.get('path'), 'r', self.logger).get_fp()
            infp.close()
            #if not os.path.exists(finfo.get('path', '')):
            #    raise Exception(unicode('input path %s does not exist' % finfo['path']).encode('utf-8'))
        #validate other required parameters
        #if not (self.user_cfg.get('ppm_x', None) and self.user_cfg.get('ppm_y', None)):
        #    raise Exception('pixels per micrometer values not provided in configuration')
        #if not self.user_cfg.get('magn', None):
        #    raise Exception('magnification value not provided in configuration')
        #if not self.user_cfg.get('type', None):
        #    raise Exception('input type not provided in configuration')
        if not self.user_cfg.get('subm_by', None):
            raise Exception('submitter user id not provided in configuration')
        if not self.user_cfg.get('partner_id', None):
            raise Exception('partner id not provided in configuration')
        if not self.user_cfg.get('set_id', None):
            raise Exception('set id not provided in configuration')
        if not (self.user_cfg.get('ctgy', None) and self.user_cfg.get('sub_ctgy', None)):
            raise Exception('input category and/or sub category not provided in configuration')

        self.logger.info('input configurations loaded and validated')

    def get_data_file_info(self, in_file, file_info):
        input_properties={}
        status = 1
        self.logger.debug("Executing DataRegistration.get_data_file_info()")
        input_properties = None
        try:
            self.logger.info("Reading the input properties for file:"+in_file)
            input_reader = baseinputreader.factory(file_info.get('type', self.user_cfg.get('type', None)), self.logger, in_file)
            input_properties = input_reader.get_properties()
            input_reader.close()
        except Exception,err:
            self.logger.exception("Unable to read the input file:"+in_file)
            return input_properties,status

        for key in ['magn', 'ppm_x', 'ppm_y', 'img_type','patch_size', 'attrib', 'global_attribs']:
            if key not in input_properties:
                if file_info.get(key, None):
                    input_properties[key] = file_info[key]
                else:
                    input_properties[key] = self.user_cfg.get(key, None)

        #override values from file_info
        #for key in ['magn', 'ppm_x', 'ppm_y']:
        #    if key in file_info:
        #        input_properties[key] = file_info[key]
        for key in file_info.iterkeys():
            input_properties[key] = file_info[key]

        status = 0
        self.logger.debug("status returned by the function DataRegistration.get_data_file_info():"+str(status))
        return input_properties,status

    def generate_registration_info(self, input_properties, in_file):
        record = {}
        partner ={}
        status = 1

        self.logger.debug("Executing DataRegistration.get_registration_info()")

        try:
            partner["_id"]= ObjectId(self.user_cfg["partner_id"])
            id,status = self.get_partner_id(partner)

            if status == 0 and id!=None:
                record["partner_id"] = id
            else:
                self.logger.error("Problem extracting the partner id:"+str(id))
                return record,status

            dt = datetime.datetime.utcnow()
            record["file_name"] = (os.path.splitext(os.path.basename(in_file))[0]).replace(str(self.mod_cfg["dataregistration"]["src_delim"]),str(self.mod_cfg["dataregistration"]["dest_delim"]))
            record["set_id"] = self.user_cfg.get("set_id", record["file_name"])
            record["file_ext"]= os.path.splitext(os.path.basename(in_file))[1]
            record["file_size"] = os.path.getsize(in_file)

            record["file_type"] = input_properties.get("type", self.user_cfg.get("type", None))
            record["desc"] = input_properties.get("desc", self.user_cfg.get("desc", ""))
            record["ctgy"] = input_properties.get("ctgy", self.user_cfg.get("ctgy", ""))
            record["sub_ctgy"] = input_properties.get("sub_ctgy", self.user_cfg.get("sub_ctgy", ""))
            record["magn"] = input_properties.get("magn", self.user_cfg.get("magn", ""))
            record["res_x"] = input_properties.get("res_x", self.user_cfg.get("res_x", None))
            record["res_y"] = input_properties.get("res_y", self.user_cfg.get("res_y", None))
            record["ppm_x"] = input_properties.get("ppm_x", self.user_cfg.get("ppm_x", None))
            record["ppm_y"] = input_properties.get("ppm_y", self.user_cfg.get("ppm_y", None))

            #image specific
            if record['file_type'] == 'image':
                record["img_type"] = input_properties.get("img_type", self.user_cfg.get("img_type", None))
                record["patch_size"] = input_properties.get("patch_size", self.user_cfg.get("patch_size", None))
                record["attrib"] = input_properties.get("attrib", self.user_cfg.get("attrib", None))
                record["global_attribs"] = input_properties.get("global_attribs", self.user_cfg.get("global_attribs", None))

            #add residual keys
            for key in input_properties.iterkeys():
                if key not in record:
                    record[key] = input_properties[key]

            record["register_ts"] = dt
            record["updt_ts"] = dt
            record["subm_by"]= ObjectId(self.user_cfg["subm_by"])
            record["file_dest"]= os.path.join(self.sys_cfg["paths"]["storage"], str(record["partner_id"]), str(dt.strftime("%Y%m%d%H")))
            record["status"] = "processing"
            record["instance_id"] = self.sys_cfg.get("instance_id", "")
            if "input_data" in self.sys_cfg['sync_entities']:
                record["sync_status"] = 'sync_pending:%s' % self.sys_cfg['instance_id']
            else:
                record["sync_status"] = "not_synced"
            #add analysis id if present
            if 'analysis_id' in self.user_cfg:
                record['analysis_id'] = ObjectId(self.user_cfg['analysis_id'])
            #fail if ppm info not provided
            if 'ppm_x' not in record or 'ppm_y' not in record:
                raise Exception('no pixels per micron value provided for input: %s' % record['file_name'])
            status = 0
        except Exception,err:
            status = 1
            self.logger.exception("Unable to create the record,"+"Error Message:"+str(sys.exc_info()[0]))
            return record,status

        self.logger.debug("status returned by the function DataRegistration.get_registration_info():"+str(status))
        return record,status

    def save_registration_info(self,record):

        status = 1

        self.logger.debug("Executing DataRegistration.get_registration_info()")

        id,status = self.mdd.post_data(self.sys_cfg["db"]["database"],self.sys_cfg["db"]["tables"]["input"],record)

        self.logger.debug("status returned by the function DataRegistration.save_registration_info():"+str(status))

        return status,id

    def store_registered_file(self,path,dest_file,in_file):
        status = 1

        self.logger.debug("Executing DataRegistration.store_registered_file()")
        self.logger.info("moving data file %s to repository" % in_file)

        if not os.path.exists(path):
            os.makedirs(path)
        shutil.copyfile(in_file, os.path.join(path, dest_file))
        status = 0

        if status == 0:
            os.remove(in_file)
            status = 0
        else:
            status = 1

        if status == 0:
            self.logger.info("move successful")
        self.logger.debug("status returned by the function DataRegistration.store_registered_file():"+str(status))

        return status

    def get_partner_id(self, query):
        id = None
        status = 1
        database = self.sys_cfg["db"]["database"]
        table = self.sys_cfg["db"]["tables"]["partners"]

        self.logger.debug("Executing DataRegistration.get_partner_id()")
        self.logger.debug("Query to get the partner_id:"+str(query))

        partner,status = self.mdd.get_data(database,table,query)

        if status == 1:
            return id,status
        else:
            for doc in partner:
                if doc["_id"]!=None:
                    status = 0
                    id = doc["_id"]
                else:
                    status = 1

                self.logger.debug("id returned by the function DataRegistration.get_partner_id():"+str(doc["_id"]))
                self.logger.debug("status returned by the function DataRegistration.get_partner_id():"+str(status))

            return id,status

    def register_data(self):
        self.logger.debug("Pipeline.invoke_registration is being invoked")

        #iterate over files in input configuration and register each of them
        for file_info in self.user_cfg.get('file_list', []):
            in_file = file_info['path']

            self.logger.info('Processing input file: %s' % in_file)


            if os.path.getsize(in_file) == 0:
                self.logger.info('Zero size file found, ignoring the file: %s' % in_file)
                continue

            input_prop,status = self.get_data_file_info(in_file, file_info)

            if status != 0:
                msg = "Error encountered while registering the in_file: %s" % in_file
                raise Exception(msg.encode('utf-8'))

            record,status = self.generate_registration_info(input_prop, in_file)
            if status != 0:
                msg = "Error encountered while registering the in_file: %s" % in_file
                raise Exception(msg.encode('utf-8'))

            status,rec_id = self.save_registration_info(record)
            if status != 0:
                msg = "Error encountered while registering the in_file: %s" % in_file
                raise Exception(msg.encode('utf-8'))

            final_path = ''
            dest_file = '%s%s' % (record["file_name"], record["file_ext"])
            final_path = os.path.join(record['file_dest'], dest_file)

            status = self.store_registered_file(record["file_dest"], dest_file, in_file)
            self.logger.info("Data in_file %s successfully registered" % in_file)


            #add sync table entry
            #update status in input control table
            num_updated,status = self.mdd.update_data(self.sys_cfg['db']['database'], self.sys_cfg['db']['tables']['input'], {'_id': rec_id}, {'$set': {'status':'success', 'updt_ts': datetime.datetime.utcnow()}})
            if status:
                msg = 'Error while updating registed in_file status'
                raise Exception(msg.encode('utf-8'))
            if "input_data" in self.sys_cfg['sync_entities']:
                sync_record = {'instance_id': self.sys_cfg.get('instance_id', ''), 'sync_type': 'input_data', 'sync_paths': [final_path], 'status': 'pending', 'source_table': self.sys_cfg['db']['tables']['input'], 'source_key': str(rec_id), 'ts': datetime.datetime.utcnow()}
                syncid,status = self.mdd.post_data(self.sys_cfg["db"]["database"], self.sys_cfg["db"]["tables"]["sync"], sync_record)
                if status:
                    msg = 'Could not schedule sync of %s with remote data store' % final_path
                    raise Exception(msg.encode('utf-8'))
            self.logger.info("Processing of %s completed, scheduled for sync with id %s" % (in_file, str(syncid)))
        #delete purge directory if given
        purge_dir = self.user_cfg.get('purge_dir', '')
        if purge_dir and os.path.exists(purge_dir):
            #check if path is a directory
            if os.path.isdir(purge_dir):
                try:
                    os.rmdir(purge_dir)
                    self.logger.info('successfully deleted purge dir %s' % (purge_dir))
                    #delete from remote store as well
                    if self.sys_cfg.get('s3_data_bucket', ''):
                        self.logger.info('deleting purge dir %s from remote store' % purge_dir)
                        s3client = boto3.client('s3')
                        objs = s3client.list_objects(Bucket=self.sys_cfg['s3_data_bucket'], Prefix=os.path.join(self.sys_cfg.get('s3_data_base_key', ''), purge_dir[1:]))
                        for obj in objs.get('Contents', []):
                            self.logger.info('deleting object %s from remote store' % obj['Key'])
                            s3client.delete_object(Bucket=self.sys_cfg['s3_data_bucket'], Key=obj['Key'])
                        self.logger.info('successfully deleted purge dir %s from remote store' % purge_dir)
                except OSError:
                    self.logger.error('could not purge dir %s as it is not empty' % (purge_dir))
                except Exception, e:
                    self.logger.error('exception while purging dir %s: %s' % (purge_dir, str(e)))
            else:
                self.logger.info('provided purge path (%s) is not a directory, skipping' % (purge_dir))
        else:
            self.logger.info('invalid or non-existant purge_dir (%s) provided, skipping' % (purge_dir))
