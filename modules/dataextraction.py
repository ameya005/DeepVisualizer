#!/usr/bin/env python
''' dataextraction.py: to extract data/patches from the input images'''

__author__ = "Rohit K. Pandey"
__copyright__ = "Copyright 2015, SigTuple Technologies Pvt. Ltd"
__version__ = "0.1"
__email__ = "rohit@sigtuple.com"

import json
from datetime import datetime
import os
from modules.mdd import MDD
from bson import ObjectId
from modules.extraction import cell
import unicodedata
import cv2
import itertools
import numpy
from modules.utils.fsutils import DataFile
from modules.inputreaders import baseinputreader


class DataExtraction(object):
    
    def __init__(self,sysys_cfg,mod_cfg,usr_cfg,lgr):
        self.logger = lgr
        
        self.logger.info("Initializing the data extraction process")
        
        self.sys_cfg = sysys_cfg
        self.mod_cfg = mod_cfg
        self.user_cfg = usr_cfg
        self.extracted_paths = {}
        #initialize DB connection
        self.mdd = MDD(self.sys_cfg['db'], self.logger)
        if not self.mdd:
            raise Exception('could not open connection to db')
        #load labels to be extracted
        self.ext_labels = self.user_cfg.get('extraction_label', {})
        self.extractors = {}
        #load extraction classes
        for ctype,cinfo in self.ext_labels.iteritems():
            self.extractors[ctype] = cell.factory(ctype, self.mod_cfg["dataextraction"].get(ctype, {}), ctype, self.user_cfg.get('additional_attrib', False), cinfo['ppm'], self.logger)
    

    #required to normalise any db special characters in query json keys
    def unicode_normalise_query(self, q):
        newq = {}
        for k,v in q.iteritems():
            if type(v) is dict:
                v = self.unicode_normalise_query(v)
            newq[unicodedata.normalize('NFKC', unicode(k))] = v
        return newq
        
    def get_files_for_extraction(self):
        
        status = 1
        file_list = []
        query = None
        query_ex = {}
        
        query_ex = self.user_cfg["query"]
        query_ex = self.unicode_normalise_query(query_ex)

        output,status = self.mdd.get_data(self.sys_cfg["db"]["database"],self.sys_cfg["db"]["tables"]["input"],query_ex)
        
        self.logger.debug("Output status of the mongogb query:"+str(status))
        
        for f in output:
            #ignore files if status not success or sync_pending on remote instance
            input_status = f.get('status', '')
            sync_status = f.get('sync_status', '')
            file_instance_id = f.get('instance_id', None)
            if input_status != 'success' or ( sync_status not in ['sync_pending:%s' % self.sys_cfg.get('instance_id', ''), "success"] and file_instance_id != self.sys_cfg.get('instance_id', '')):
                self.logger.info('Ignoring file %s%s with status %s and sync_status %s for extraction' %(f['file_name'],f['file_ext'],input_status, sync_status))
                continue
            file_path = os.path.join(f["file_dest"], '%s%s' % (f["file_name"], f["file_ext"]))
            file_list.append({"id": f["_id"], "path": file_path, "type": f["file_type"], "ppm_x": f["ppm_x"], "ppm_y": f["ppm_y"], "patch_size":f.get("patch_size", 0), "img_type":f.get("img_type", "raw"), "attrib":f.get("attrib", {}), "global_attribs": f.get("global_attribs", {})})
            status = 0
            
        self.logger.debug("File list size generated from the mongodb query:"+str(len(file_list)))
        self.logger.debug("Status returned by the function DataExtraction.get_files_for_extraction():"+str(status))

        #error if no files found
        if len(file_list) == 0:
            raise Exception('no valid input files found for extraction')
        
        return file_list,status
        
    def init_data_extraction(self,files_list):
        status = 1
        record = {}
        id = None
        
        self.logger.debug("Executing the function DataExtraction.init_dataextraction()")
       
        record["last_updt_dt_tm"]=datetime.utcnow()
        record["updated_by"]=self.user_cfg["updated_by"]
        record["config_ext"]=self.mod_cfg
        record["file_info"] = files_list
        record["write_images"] = self.user_cfg.get("write_images", False)
        record["write_centroids"] = self.user_cfg.get("write_centroids", False)
        record["additional_attrib"] = self.user_cfg.get("additional_attrib", False)
        record["all_global_attrib"] = self.user_cfg.get("all_global_attrib", False)
        self.all_global_attrib = record["all_global_attrib"]

        if 'analysis_id' in self.user_cfg:
            record['analysis_id'] = ObjectId(self.user_cfg['analysis_id'])
        
        database = self.sys_cfg["db"]["database"]
        collection = self.sys_cfg["db"]["tables"]["extraction"]
        
        record["status"] = "initialised"
        record["instance_id"] = self.sys_cfg.get("instance_id", "")
        if "extraction_output" in self.sys_cfg['sync_entities']:
            record["sync_status"] = "sync_pending:" + self.sys_cfg.get("instance_id", "")
        else:
            record["sync_status"] = "not_synced"
        id,status = self.mdd.post_data(database,collection,record)
        ext_path = self.sys_cfg["paths"]["extraction"]
        self.extraction_id = id
        self.img_path = os.path.join(ext_path, str(id), "images")
        self.cent_path = os.path.join(ext_path, str(id), "centroids")
        self.attrib_path = os.path.join(ext_path,str(id), "attrib")
        self.logger.debug("Status returned by the function DataExtraction.init_dataextraction():"+str(status))
       
        return record,id,status
        
    def invoke_data_extraction(self,record):
        
        status = 1
        
        self.logger.debug("Invoking the function DataExtraction.invoke_dataextraction()")

        for ext_label, ext_info in self.ext_labels.iteritems():
            extractor = self.extractors.get(ext_label, None)
            total_count = ext_info.get('count', 0)
            count = 0
            out_ppm = ext_info.get('ppm', [])
            if not out_ppm:
                raise Exception('no ppm information provided for extraction label %s' % ext_label)
            if not extractor:
                raise Exception('no extractor object provided for extraction label %s' % ext_label)

            #add empty path entries in extracted paths for downstream modules
            self.extracted_paths.setdefault(ext_label, {})['images'] = ''
            self.extracted_paths.setdefault(ext_label, {})['centroids'] = ''
            self.extracted_paths.setdefault(ext_label, {})['attrib'] = ''
            self.extracted_paths.setdefault(ext_label, {})['global_attrib'] = ''

            self.logger.info(self.extracted_paths)

            input_size = 0
            for rec in record["file_info"]:
                if count >= total_count and not self.all_global_attrib:
                    break
                img_type = rec.get('img_type', 'raw')
                input_size += DataFile(rec['path'], 'r',self.logger).get_size()
                if img_type == ext_label:
                    self.logger.info('Processing grid image  %s for extraction of %s' % (rec['path'], ext_label))
                    (img_patches, attribs) = self.extract_patches_from_grid_image(rec,out_ppm)
                elif img_type == 'raw':
                    self.logger.info('Processing image  %s for extraction of %s' % (rec['path'], ext_label))
                    (img_patches, attribs) = extractor.extract_patches(rec, (total_count-count), all_global_attrib=self.all_global_attrib)
                else:
                    self.logger.info("Ignoring image %s, unknown img_type: %s" % (rec['path'], img_type))
                    continue
                if len(img_patches) > (total_count-count):
                    img_patches = {k:img_patches[k] for k in itertools.islice(img_patches,0,(total_count-count))}
                #write out patches
                self.write_data_file(ext_label, img_patches, attribs, out_ppm)
                count += len(img_patches)
            self.logger.info('for label %s, extracted: %d, required: %d' % (ext_label, count, total_count))


        record["last_updt_dt_tm"] = datetime.utcnow()
        record['extraction_path'] = os.path.join(self.sys_cfg["paths"]["extraction"], str(self.extraction_id))
        return record,status,input_size
    
    def update_data_record(self,record):
        
        self.logger.debug("Starting the record updated. Invoked function DataExtraction.update_data_record()")
        database = self.sys_cfg["db"]["database"]
        collection = self.sys_cfg["db"]["tables"]["extraction"]

        self.logger.debug("Updating table with database:"+database+",collection:"+collection)
        record['output'] = self.extracted_paths
        (num_updated,status) = self.mdd.update_data(database,collection,{'_id':self.extraction_id},{'$set':{'status':'success','output': self.extracted_paths,'last_updt_dt_tm': datetime.utcnow()}})
        if status == 0:
            self.logger.info("Id returned after successful update of the record:" +str(id))
        else:
            self.logger.exception("Error encountered while updating the mongodb records")
        return status

    def schedule_for_sync(self):
        self.logger.debug("Scheduling the extraction for sync. Invoked function DataExtraction.schedule_for_sync")
        database = self.sys_cfg["db"]["database"]
        collection = self.sys_cfg["db"]["tables"]["sync"]

        sync_record = {}
        sync_record['instance_id'] = self.sys_cfg.get('instance_id', '')
        sync_record['ts'] = datetime.utcnow()
        sync_record['status'] = 'pending'
        sync_record['sync_type'] = 'extraction'
        sync_record['source_table'] = 'extraction_control'
        sync_record['source_key'] = str(self.extraction_id)

        sync_paths = []
        for cell_type, paths in self.extracted_paths.iteritems():
            for path_type, path in paths.iteritems():
                if path_type == "images":
                    continue
                if os.path.exists(path):
                    sync_paths.append(path)

        sync_record['sync_paths'] = sync_paths
        (id, status) = self.mdd.post_data(database, collection, sync_record)

        if status:
            raise Exception("Failed to schedule the extraction for sync.")

    def write_data_file(self, label, image_patches, attribs, out_ppm):
        count_centroids = 0
        count_images = 0
        image_flag = self.user_cfg.get("write_images", False)
        centroid_flag = self.user_cfg.get("write_centroids", False)
        height = 2 * int(self.mod_cfg["dataextraction"][label]["out_patch_height_um"]*out_ppm[1])
        width = 2 * int(self.mod_cfg["dataextraction"][label]["out_patch_width_um"]*out_ppm[0])
        img_path = os.path.join(self.img_path, label)
        cent_path = os.path.join(self.cent_path, label)
        attrib_path = os.path.join(self.attrib_path,label)
        
        additional_attrib_flg = self.user_cfg.get("additional_attrib", False)

        self.logger.debug("Writing the images for "+label+"on the path:"+img_path)
        self.logger.debug("Writing the centroids for "+label+"on the path:"+cent_path)
        
        if not os.path.exists(img_path) and image_flag == 1:
            os.makedirs(img_path)
            
        if not os.path.exists(cent_path) and centroid_flag == 1:
            os.makedirs(cent_path)
        if image_flag == 1:
            self.extracted_paths.setdefault(label, {})['images'] = img_path
        if centroid_flag == 1:
            centroid_file_path = os.path.join(cent_path, "centroids.txt")
            self.extracted_paths.setdefault(label, {})['centroids'] = centroid_file_path
            f = DataFile(centroid_file_path, "a", self.logger).get_fp()
        
        if additional_attrib_flg:
            if not os.path.exists(attrib_path):
                os.makedirs(attrib_path)
            attrib_file_path = os.path.join(attrib_path,"attrib.txt")
            self.extracted_paths.setdefault(label, {})['attrib'] = attrib_file_path
            f_attrib = DataFile(attrib_file_path, "a", self.logger).get_fp()

        for key,value in image_patches.iteritems():
            img_name = '%s.jpg' % key
            if height == value.shape[1] and width == value.shape[0]:
                if image_flag == 1:
                    cv2.imwrite(os.path.join(img_path,img_name).encode("utf-8"),value)
                    count_images += 1
                if centroid_flag == 1:
                    f.write(key+"\n")
                    count_centroids +=1
                if additional_attrib_flg:
                    out_val = {}
                    out_val['name'] = img_name
                    for attrib_name, attrib_vals in attribs.iteritems():
                        if key in attrib_vals:
                            out_val[attrib_name] = attrib_vals[key]
                    if len(out_val) > 1:
                        f_attrib.write('%s\n' % (json.dumps(out_val, ensure_ascii=False, encoding='utf-8')))
            else:
                self.logger.error("Image size less than the patch height,width:"+ str(value.shape[0]) + "," + str(value.shape[1]))
        #add global attributes to files
        if additional_attrib_flg and 'global' in attribs:
            self.extracted_paths.setdefault(label, {})['global_attrib'] = os.path.join(self.attrib_path, label, "global_attrib.txt")
            self.write_global_attributes(label, attribs['global'])

        if centroid_flag == 1:        
            f.close()
        if additional_attrib_flg:
            f_attrib.close()
        return count_images,count_centroids

    '''
    Write global attributes for the label to file. attribs can be list of dict(in case of grid image) as well as a single dict (for raw image).
    '''
    def write_global_attributes(self, label, attribs):
        global_attrib_file_path = os.path.join(self.attrib_path, label, "global_attrib.txt")
        f_global_attrib = DataFile(global_attrib_file_path, "a", self.logger).get_fp()
        if isinstance(attribs, list):
            for entry in attribs:
                f_global_attrib.write('%s\n' % (json.dumps(entry, ensure_ascii=False, encoding='utf-8')))
        else:
            f_global_attrib.write('%s\n' % (json.dumps(attribs, ensure_ascii=False, encoding='utf-8')))
        f_global_attrib.close()




    #return extracted paths for use by other modules
    def get_extraction_paths(self):
        return self.extracted_paths

    def extract_patches_from_grid_image(self,input_info,out_ppm):
        input_reader = baseinputreader.factory(input_info['type'], self.logger, inpath = input_info['path'], in_ppm=[input_info['ppm_x'], input_info['ppm_y']], patch_size=input_info['patch_size'], jump=input_info['patch_size'], out_ppm=out_ppm)
        image_patches = {}
        ind = 0
        try:
            for out_y,out_x,image_patch in input_reader:
                # Check if returned patch is all black
                # Lower and upper RGB values for black: [0,0,0] and [20,10,10]
                mask = cv2.inRange(image_patch, numpy.array([0,0,0]), numpy.array([20,10,10]))
                if((numpy.size(mask) - numpy.count_nonzero(mask)) > 0):
                    image_patches[input_info['attrib'][ind]['name']] = image_patch
                    ind = ind + 1
        except:
            pass

        '''
        Structure of per cell attributes stored in mongodb:-

        attrib: [
        {
            name: IMG-19_rbc_0_0,
            diameter: 12,
            rgb: { red: 1, blue: 1, green: 1},
            ratio: 11,
            area: 11
        },
        ...
        ]

        Structure expected here:-

        attrib: {
            diameter: {
                0_0: 12,
                ...
            },
            area: {
                0_0: 11,
                ...
            },
            ratio: {
                0_0: 11,
                ...
            },
            rgb: {
                0_0: {red: 1, blue: 1, green: 1},
                ...
            }

        }
        '''
        out_attribs = {'diameter': {}, 'area': {}, 'ratio': {}, 'rgb': {}}
        for entry in input_info['attrib']:
            out_attribs['diameter'][entry['name']] = entry['diameter']
            out_attribs['area'][entry['name']] = entry['area']
            out_attribs['ratio'][entry['name']] = entry['ratio']
            out_attribs['rgb'][entry['name']] = entry['rgb']
        out_attribs['global'] = input_info['global_attribs']
        return (image_patches, out_attribs)


 
def invoke_data_extraction(sysys_cfg,mod_cfg,usr_cfg, logger):
    '''Read the config file'''
    
    de = DataExtraction(sysys_cfg,mod_cfg,usr_cfg,logger)

    files_list,status = de.get_files_for_extraction()
    
    if status == 0:
        record,id,status = de.init_data_extraction(files_list)
    else:
        raise Exception("DataRegistration.init_data_extraction function failed")
        
    if status == 0 and id!=None:
        (record,status,input_size) = de.invoke_data_extraction(record)
    else:
        raise Exception("DataExtraction.invoke_data_extraction() failed, check the file name")
        
    status = de.update_data_record(record)
    if status:
        raise Exception('could not update extraction status in db')

    logger.info('extraction completed')
    record['_id'] = id

    record['no_of_files'] = len(files_list)
    record['extraction_size'] = DataFile(record['extraction_path'], 'r', logger, folder=True).get_size()
    record['input_size'] = input_size

    if "extraction_output" in de.sys_cfg['sync_entities']:
        de.schedule_for_sync()

    return record
 
