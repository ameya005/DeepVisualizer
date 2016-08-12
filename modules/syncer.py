#!/usr/bin/env python
'''
This is a sync-daemon utility.
Source of truth for it will be a mongodb table where each slave will update local files which it needs to sync to master/S3.

One instance should run on each slave, taking up these tasks in the background and updating status in respective tables on success/failure.

Types of data files to sync

|------------------+--------------------------+----------------------------------|
| Name             | Associated Control Table | File Type                        |
|------------------+--------------------------+----------------------------------|
| Registered Input | input_control            | File (slide/image)               |
|------------------+--------------------------+----------------------------------|
| Training Data    | train_data_control       | Dir (containing annotation file) |
|------------------+--------------------------+----------------------------------|
| Model            | mmodel_version_control   | File (pickled model file)        |
|------------------+--------------------------+----------------------------------|
| Analysis         | analysis_control         | Dir (containing analysis output) |
|------------------+--------------------------+----------------------------------|
| Extraction       | extraction_control       | Dir(containing extraction output)|
|------------------+--------------------------+----------------------------------|
'''

import os,sys
import json
import time
import glob
from datetime import datetime
from bson.objectid import ObjectId
from mdd import MDD
import boto3
import s3transfer

class Syncer(object):

    def __init__(self, sys_config, module_config, logger):

        self.logger = logger
        self.sys_config = sys_config
        self.module_config = module_config
        self.db_config = self.sys_config.get('db', {})

        #open db connection
        self.dbclient = MDD(self.db_config, self.logger)
        if not self.dbclient:
            raise Exception('could not initialise db connection')
        self.db = self.db_config['database']

        self.sync_control_table = self.db_config['tables']['sync']
        self.input_control_table = self.db_config['tables']['input']
        self.train_table = self.db_config['tables']['train_data']
        self.extraction_table = self.db_config['tables']['extraction']
        self.model_control_table = self.db_config['tables']['model_version']

        self.lock_path = self.sys_config['paths']['sync']
        self.instance_id = self.sys_config['instance_id']
        self.global_pid_lock = os.path.join(self.lock_path,'_sync.pid')

        self.s3_bucket = self.sys_config.get('s3_data_bucket')
        self.s3_base_key = self.sys_config.get('s3_data_base_key')

        if not (self.s3_bucket or self.s3_base_key):
            raise Exception('remote data store details not provided in system global config')

        if self.__is_locked(self.global_pid_lock):
            self.logger.error("Aborting Process. There is a global lock on the sync process. A lock file %s exists with an active PID" % self.global_pid_lock)
            exit(1)

    def run_sync(self):
        while True:
            #get sync information from the database
            (sync_info, status) = self.dbclient.get_one(self.db, self.sync_control_table, {'instance_id': self.instance_id,'status' : 'pending'})
            if status:
                raise Exception('could not fetch info from db for any pending sync status')
    
            if None is sync_info:
                #do nothing
                self.logger.info("no pending sync rows")
                return
    
            self.logger.info("sync_info = %s" % str(sync_info))
    
            #get all sync info
            sync_id = sync_info['_id']  #getting the sync_id
            sync_paths = sync_info['sync_paths'] #list of paths to be synced
            sync_type  = sync_info['sync_type'] #type of sync {input_data | train_data | model}
            sync_source_table = sync_info['source_table'] #table that needs to be updated on success or failure
            sync_source_id = sync_info['source_key'] # key that needs to be updated on success or failure
            sync_source_status_key = sync_info.get('source_status_key', '') #status key
            sync_source_status_value = sync_info.get('source_status_value', '') #status value
            lock_path = os.path.join(self.lock_path,'%s.lock' % (sync_id))
            #check if the
            if self.__is_locked(lock_path):
                self.logger.info("lock_path %s is locked" % (lock_path))
                continue
                #self.run_sync()
    
            try:
                self.__db_update_status(sync_id,'sync_initiated',self.sync_control_table)
                self.logger.info('Updated the status of sync_id : %s to sync_initiated' % str(sync_id))
                #fetch resource from S3
                s3 = boto3.client('s3')
                transfer = s3transfer.S3Transfer(s3)
    
                for path in sync_paths:
                    path = os.path.abspath(path)
                    files_to_upload = []
    
                    if not os.path.exists(path):
                        self.logger.error("File %s not present on local file system" % path)
                        raise Exception('File not present on local file system')
                    elif os.path.isdir(path):
                        for (src_dir, dirs, files) in os.walk(path):
                            for file in files:
                                files_to_upload.append(os.path.join(src_dir, file))
                    else:
                        files_to_upload.append(path)
    
                    for file_path in files_to_upload:
                        file_path = os.path.abspath(file_path)
                        #remove leading '/'
                        s3_key = file_path[1:]
                        s3_key = os.path.join(self.s3_base_key, s3_key)
                        self.logger.info('starting file upload of %s from s3 bucket: %s and key: %s' % (file_path,self.s3_bucket, s3_key))

                        transfer.upload_file(file_path,self.s3_bucket,s3_key)

                        self.logger.info('s3 upload completed')
    
    
                os.remove(lock_path)
                self.__db_update_status(sync_id,'success',self.sync_control_table)
                #update default sync status 
                self.__db_update_status(sync_source_id,'success',sync_source_table,status_key='sync_status',ts_key='sync_ts')
                #update host table status if provided
                if sync_source_status_key and sync_source_status_value:
                    self.__db_update_status(sync_source_id,sync_source_status_value,sync_source_table,status_key=sync_source_status_key,ts_key='sync_ts')
            except Exception as e:
                self.logger.exception(str(e))
                self.__db_update_status(sync_id,'failed',self.sync_control_table)
                os.remove(lock_path)
            self.logger.info('sync complete for sync_id = %s' % str(sync_id))
            #self.run_sync()

    '''
    A simple util to update the status of the sync table.
    '''
    def __db_update_status(self,_id,status,table,status_key='status',ts_key='ts'):
        (sync_id, ret_status) = self.dbclient.update_data(self.db, table , {'_id': ObjectId(_id)},{'$set' : { status_key: str(status), ts_key: datetime.utcnow()}})
        if ret_status:
            if 'failed' is not status:
                self.__db_update_status(_id,'failed')
            raise Exception('could not update status in %s table for sync_id %s' % (table,_id))


    '''
    checks if the data sync is already captured by another process.
    if the lock exists, it looks into the lock file to get the PID of the process that initiated the sync,
    it checks if the PID is running, if yes. then - it does nothing, else, it removes the lock file and reinitiates the sync.
    '''
    def __is_locked(self,lock_path):
        try:
            fd = os.open(lock_path, os.O_WRONLY|os.O_CREAT|os.O_EXCL)
            os.write(fd,str(os.getpid()))
            os.close(fd)
            return False
        except:
            self.logger.info('could not open lock file %s' % (lock_path))
            try:
                with open(lock_path) as f:
                    pid = f.readline()
                    os.kill(int(pid),0)
                    return True
            except OSError:
                self.logger.info('Lock file present, but, the PID is not')
                os.remove(lock_path)
                return self.__is_locked(lock_path)
            except Exception as e:
                self.logger.exception('Exception in handling lock in sync.')
                return True

