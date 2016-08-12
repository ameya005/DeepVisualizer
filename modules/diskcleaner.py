from mdd import MDD
import os
import shutil
from bson import ObjectId


class DiskCleaner(object):

    def __init__(self, sys_config, logger):

        self.logger = logger
        self.sys_config = sys_config
        self.db_config = self.sys_config.get('db', {})
        self.cleanup_config = self.sys_config.get("cleanup", {})

        #open db connection
        self.dbclient = MDD(self.db_config, self.logger)
        if not self.dbclient:
            raise Exception('could not initialise db connection')
        self.db = self.db_config['database']

        self.input_control_table = self.db_config['tables']['input']
        self.extraction_table = self.db_config['tables']['extraction']
        self.analysis_table = self.db_config['tables']['analysis']


    def run_cleanup(self):
        # Delete files in storage if synced and associated analysis is done if there is any
        for (src_dir, dirs, files) in os.walk(self.cleanup_config.get('paths',{}).get('storage', '')):
            for f in files:
                record,status = self.dbclient.get_one(self.db, self.input_control_table, {"file_name": os.path.splitext(f)[0]})
                if record and not status and record.get('sync_status', '') in ["success","not_synced"]:
                    if record.get('analysis_id', None):
                        analysis_record,status = self.dbclient.get_one(self.db, self.analysis_table, {"_id": ObjectId(record['analysis_id'])})
                        if analysis_record and not status and analysis_record.get("status", "") not in ['success', 'failure']:
                            self.logger.debug("Not deleting %s, analysis not complete" % f)
                            continue
                    # On Windows, attempting to remove a file that is in use causes an exception to be raised;
                    # on Unix, the directory entry is removed but the storage allocated to the file is not made available until the original file is no longer in use.
                    self.logger.info("Deleting file: %s/%s" % (src_dir,f))
                    os.remove(os.path.join(src_dir, f))

        # Delete files in extraction if synced and associated analysis is done if there is any
        extraction_id = None
        for (src_dir, dirs, files) in os.walk(self.cleanup_config.get('paths',{}).get('extraction', '')):
            if src_dir == self.sys_config['paths']['extraction']:
                continue
            # Assumptions: extraction directory name is extraction_id
            # Directory is being traversed top down
            if not extraction_id or extraction_id not in src_dir:
                extraction_id = os.path.basename(os.path.abspath(src_dir))

            record,status = self.dbclient.get_one(self.db, self.extraction_table, {"_id": ObjectId(extraction_id)})
            if record and not status and record.get('sync_status', '') in ["success","not_synced"]:
                if record.get('analysis_id', None):
                    analysis_record,status = self.dbclient.get_one(self.db, self.analysis_table, {"_id": ObjectId(record['analysis_id'])})
                    if analysis_record and not status and analysis_record.get("status", "") not in ['success', 'failure']:
                        self.logger.debug("Not removing extraction %s, analysis %s not complete" %(extraction_id, record['analysis_id']))
                        continue
                self.logger.info("Removing dir: %s" % src_dir)
                shutil.rmtree(src_dir, ignore_errors=True)


        # Delete analysis files if analysis data has been synced.
        analysis_id = None
        for (src_dir, dirs, files) in os.walk(self.cleanup_config.get('paths',{}).get('analysis','')):
            if src_dir == self.cleanup_config.get('paths',{})['analysis']:
                continue
            # Assumption: analysis directory name is analysis_id
            # Directory is being traversed top down
            if not analysis_id or analysis_id not in src_dir:
                analysis_id = os.path.basename(os.path.abspath(src_dir))
            record,status = self.dbclient.get_one(self.db, self.analysis_table, {"_id": ObjectId(analysis_id)})
            if record and not status and record.get('sync_status', '') in ["success","not_synced"]:
                self.logger.info("Removing dir: %s" % src_dir)
                shutil.rmtree(src_dir, ignore_errors=True)
        self.logger.info('Cleanup completed')
