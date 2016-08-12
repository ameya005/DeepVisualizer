#!/usr/bin/env python

import os,sys
import codecs
import boto3
import json

#returns sys config file loaded into json object 
def get_system_config(config_type):
    config_base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'config')
    if config_type == 'sys_globals':
        file_name = 'sys_globals.json'
    elif config_type ==  'module_config':
        file_name = 'mod_config.json'
    else:
        raise Exception('unknown system config type %s' % config_type)

    sys_config = {}
    gf = open(os.path.join(config_base_path, file_name), 'r')
    sys_config = json.load(gf)
    gf.close()
    return sys_config


#Fetches file from S3 if not available in local FS.
#Creates a lock file so that multiple process instances
#do not try to fetch the same file.
class DataFile(object):

    def __init__(self, file_path, mode, logger, is_binary=False, sort=False, folder=False):
        self.logger = logger
        self.file_path = os.path.abspath(file_path)
        self.mode = mode
        self.is_binary = is_binary
        self.folder = folder

        if self.folder:
            self.mode = 'r'
            self.is_binary = False
            self.sort = False

        self.fp = None
        
        if not self.file_path:
            raise Exception('empty file key provided')

        self.lock_path = '%s.lock' % (self.file_path)
        is_local_exists = False

        if 'r' not in self.mode:
            #create parent dir if reqd
            parent_dir = os.path.dirname(self.file_path)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            is_local_exists = True
        elif os.path.exists(self.file_path) and not os.path.exists(self.lock_path):
            is_local_exists = True
        else:
            self.logger.info('file %s not present on local file system, trying remote data store' % self.file_path)
            self.fetch_remote_data_object()
            is_local_exists = True

        if is_local_exists:
            if sort:
                self.fp = codecs.open(self.file_path, "r+",'utf-8')
                self.sort()
                self.fp.close()

            if self.is_binary:
                self.fp = open(self.file_path, self.mode)
            elif not self.folder:
                self.fp = codecs.open(self.file_path, self.mode, 'utf-8')
        else:
            raise Exception(unicode('could not find file path %s' % (self.file_path)).encode('utf-8'))

    def get_fp(self):
        if self.folder:
            raise Exception("can't get fp for a folder")
        return self.fp


    def fetch_remote_data_object(self):
        self.sys_config = get_system_config('sys_globals')

        self.s3_bucket = self.sys_config.get('s3_data_bucket')
        self.s3_base_key = self.sys_config.get('s3_data_base_key')

        if not self.s3_bucket:
            raise Exception('remote data store details not provided in system global config')

        #create parent dir if reqd
        parent_dir = os.path.dirname(self.file_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        #create lock file
        while True:
            try:
                fd = os.open(self.lock_path, os.O_WRONLY|os.O_CREAT|os.O_EXCL)
                os.write(fd, str(os.getpid()))
                os.close(fd)
                break
            except:
                #check if prev process is still running, otherwise delete lock file and try again
                lfp = open(self.lock_path, 'r')
                prev_pid = int(lfp.read().strip())
                lfp.close()
                is_running = False
                try:
                    os.kill(prev_pid, 0)
                except OSError:
                    is_running = False
                else:
                    is_running = True
                if is_running:
                    self.logger.exception('could not open lock file to fetch file %s from S3, another process (%d) is syncing the file' % (self.file_path, prev_pid))
                    raise
                else:
                    #delete file and open again
                    self.logger.info('process %d which opened lock file %s is no longer running, hence deleting lock file' % (prev_pid, self.lock_path))
                    os.remove(self.file_path)
                    os.remove(self.lock_path)

        try:
            #fetch from S3
            s3 = boto3.resource('s3')
            #remove leading '/'
            s3_key = self.file_path[1:]
            if self.s3_base_key:
                s3_key = os.path.join(self.s3_base_key, s3_key)
            self.logger.info('starting file download from s3 bucket: %s and key: %s' % (self.s3_bucket, s3_key))
            bucket = s3.Bucket(self.s3_bucket)
            if not self.folder:
                bucket.download_file(s3_key, self.file_path)
            else:
                for obj in bucket.objects.filter(Prefix=s3_key):
                    folders = obj.key.split('/')
                    parent_dir = os.path.dirname(os.path.join('/', *folders[1:]))
                    if not os.path.exists(parent_dir):
                        os.makedirs(parent_dir)
                    bucket.download_file(obj.key, os.path.join('/', *folders[1:]))
            self.logger.info('s3 download finished')
        finally:
            #delete lock file
            os.remove(self.lock_path)

    def sort(self):
        lines = sorted(self.fp.readlines())
        lines = filter(None, lines)
        self.fp.seek(0)
        for line in lines:
            self.fp.write("%s" % line)

    def close(self):
        if self.fp:
            self.fp.close()

    def get_size(self):
        if not self.folder:
            return os.path.getsize(self.file_path)
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self.file_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        return total_size
