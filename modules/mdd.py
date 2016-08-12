#!/usr/bin/env python
''' mdd.py: to connect and perform operations on MongoDB'''

__author__ = "Rohit K. Pandey"
__copyright__ = "Copyright 2015, SigTuple Technologies Pvt. Ltd"
__version__ = "0.1"
__email__ = "rohit@sigtuple.com"

''' This code is a wrapper on pymongo client to provide a
    seamless driver for interacting with Mongodb. The class
    expects a config file for initiation of the object. The
    config file should have relevant information about the
    connection string'''


from pymongo import MongoClient
from pymongo import ReturnDocument
from pymongo import errors

class MDD(object):

    def __init__(self,config,lgr):
        self.config = config
        self.logger = lgr
        self.client = self.get_client()

    def get_client(self):

        self.logger.debug("Invoking the MDD.get_client() function")
        self.logger.debug("Connection string for Mongo:"+str(self.config["connection_string"]))
        try:
            self.logger.info('Opening DB Connection')
            client = MongoClient(self.config["connection_string"],w=self.config.get('write_concern',1),j=self.config.get('journal',True),wtimeout=self.config.get('write_timeout',300))
            if self.config.get('user', None) and self.config.get('passwd', None):
                self.logger.info('authenticating db user')
                if not client.admin.authenticate(self.config['user'], self.config['passwd']):
                    raise Exception('could not authenticate credentials provided in configuration on db')
        except Exception,err:
            self.logger.exception("Initialization of the mongodb client failed")
            raise

        return client

    def get_one(self,database,collection,query,sort=[]):
        status = 1

        self.logger.debug("Invoking the MDD.get_one() function")
        self.logger.debug("database:"+database+", collection:"+collection+",query:"+str(query)+",sort:"+str(sort))
        out = None
        retries = 0
        while retries < self.config.get('retries',1):
            try:
                mdddb = self.client[database]
                mddcol = mdddb[collection]
                out = mddcol.find_one(query, sort=sort)
                status = 0
                break
            except errors.AutoReconnect:
                self.logger.info("Exception while executing query %s , retrying" % query)
                retries += 1
            except Exception,err:
                self.logger.exception("Execution of mongodb query failed for the following query:"+str(query))

        self.logger.debug("output of the function returned by MDD.get_one() " + str(out))

        return out,status

    def get_data(self,database,collection,query,sort=[],limit=0,projection=None):
        status = 1

        self.logger.debug("Invoking the MDD.get_data() function")
        self.logger.debug("database:"+database+", collection:"+collection+",query:"+str(query))
        retries = 0
        while retries < self.config.get('retries',1):
            try:
                mdddb = self.client[database]
                mddcol = mdddb[collection]
                out = mddcol.find(filter=query,sort=sort,limit=limit,projection=projection)
                status = 0
                break
            except errors.AutoReconnect:
                self.logger.info("Exception while executing query %s , retrying" % query)
                retries += 1
            except Exception,err:
                self.logger.exception("Execution of mongodb query failed for the following query:"+str(query))
                raise

        self.logger.debug("output of the function returned by MDD.get_data()")

        return out,status


    def post_data(self,database,collection,data_post):
        status = 1
        self.logger.debug("Invoking the MDD.post_data() function")
        self.logger.debug("database:"+database+", collection:"+collection+",data_post:"+str(data_post))

        id = None
        retries = 0
        while retries < self.config.get('retries',1):
            try:
                mdddb = self.client[database]
                posts = mdddb[collection]
                id = posts.insert_one(data_post).inserted_id
                status = 0
                break
            except errors.AutoReconnect:
                self.logger.info("Exception while posting data %s , retrying" % data_post)
                retries += 1
            except Exception, err:
                self.logger.exception("The insert data query on Mongodb failed for the query:"+ str(data_post))
                raise

        self.logger.debug("id returned by the function MDD.post_data():"+str(id))
        return id,status

    def update_data(self,database,collection,filter,data_post,upsert=False):
        status = 1
        self.logger.debug("Invoking the MDD.post_data() function")
        self.logger.debug("database:"+database+", collection:"+collection+",data_post:"+str(data_post))

        retries = 0
        while retries < self.config.get('retries',1):
            try:
                mdddb = self.client[database]
                posts = mdddb[collection]
                result = posts.update_one(filter,data_post,upsert=upsert)
                status = 0
                break
            except errors.AutoReconnect:
                self.logger.info("Exception while executing update %s on %s , retrying" % (data_post, filter))
                retries += 1
            except Exception,err:
                self.logger.exception("Update data query failed on Mongodb for the following query:"+str(data_post))
                raise

        self.logger.debug("updated records returned by the function MDD.update_data(): %d" % result.matched_count)

        return result.matched_count,status


    def find_one_and_update(self,database,collection,filter,data_post,sort=None,upsert=False):
        status = 1
        self.logger.debug("Invoking the MDD.find_one_and_update function")
        self.logger.debug("database:"+database+", collection:"+collection+",query"+str(filter)+",data_post:"+str(data_post))
        retries = 0
        while retries < self.config.get('retries',1):
            try:
                mdddb = self.client[database]
                posts = mdddb[collection]
                result = posts.find_one_and_update(filter,data_post,sort=sort,return_document=ReturnDocument.BEFORE,upsert=upsert)
                status = 0
                break
            except errors.AutoReconnect:
                self.logger.info("Exception while executing update %s on %s, retrying" % (data_post,filter))
                retries += 1
            except Exception,err:
                self.logger.exception("find_one_and_update data query failed on Mongodb for the following query:"+str(data_post))
                raise

        self.logger.debug("updated record returned by the function MDD.find_one_and_update: %s" % result)

        return result,status

    def find_distinct(self, database, collection, query, field):
        status = 1

        self.logger.debug("Invoking the MDD.find_distinct function")
        self.logger.debug("database:"+database+", collection:"+collection+",query:"+str(query)+", distinct:"+str(field))
        retries = 0
        while retries < self.config.get('retries',1):
            try:
                mdddb = self.client[database]
                mddcol = mdddb[collection]
                out = mddcol.find(query)
                if out is not None:
                    out = out.distinct(field)
                status = 0
                break
            except errors.AutoReconnect:
                self.logger.info("Exception while executing query %s , retrying" % query)
                retries += 1
            except Exception,err:
                self.logger.exception("Execution of mongodb query failed for the following query:"+str(query))
                raise

        self.logger.debug("output of the function returned by MDD.find_distinct: %s " % out)

        return out,status

    def get_aggregated_data(self, database, collection, pipeline):
        status = 1

        self.logger.debug("Invoking the MDD.get_aggregated_data function")
        self.logger.debug("database:"+database+", collection:"+collection+",query:"+str(pipeline))
        retries = 0
        while retries < self.config.get('retries',1):
            try:
                mdddb = self.client[database]
                mddcol = mdddb[collection]
                out = mddcol.aggregate(pipeline=pipeline)
                status = 0
                break
            except errors.AutoReconnect:
                self.logger.info("Exception while executing query %s , retrying" % pipeline)
                retries  += 1
            except Exception,err:
                self.logger.exception("Execution of mongodb query failed for the following query:"+str(pipeline))
                raise

        self.logger.debug("output of the function returned by MDD.get_aggregated_data: %s " % out)

        return out,status