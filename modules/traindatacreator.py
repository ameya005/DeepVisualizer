#!/usr/bin/env python

import os,sys
from optparse import OptionParser
import json
import random
import numpy
import base64
import pymongo
from collections import OrderedDict
from datetime import datetime
from mdd import MDD
from inputreaders import baseinputreader
from imagevariants import baseimagevariant
from imageprocs import baseimageproc
from modules.extraction import cell
from modules.utils.fsutils import DataFile
from modules.utils.imageparamutils import ImageParamUtils
import cv2

class TrainDataCreator(object):

    def __init__(self, sys_config, module_config, config, logger):
        self.sys_config = sys_config
        self.module_config = module_config
        self.config = config
        self.logger = logger
        self.outfps = {}
        self.db_config = sys_config.get('db', {})
        self.input_readers = OrderedDict()
        self.max_open_readers = self.module_config.get('train', {}).get('max_open_readers', 20)
        self.shuffle_keys = self.module_config.get('train', {}).get('shuffle_keys', [])
        if not self.load_and_validate_config():
            raise Exception('could not validate config for TrainingDataCreator instantiation')

    def load_and_validate_config(self):
        self.logger.info('loading and validating config')

        #check global config
        self.train_data_base_path = self.sys_config['paths'].get('train_data', '')
        if not self.check_path_exists(self.train_data_base_path):
            return False

        #check all filepaths and dirs
        self.logger.info('checking all file paths and directories in config file')
        self.data_type = self.config.get('type', '')

        self.dbclient = None

        if self.db_config:
            self.dbclient = MDD(self.db_config, self.logger)
            if not self.dbclient:
                return False

            self.db = self.db_config['database']
            self.input_control_table = self.db_config['tables']['input']
            self.train_data_table = self.db_config['tables']['train_data']
            self.model_table = self.db_config['tables']['models']
            self.sync_table = self.db_config['tables']['sync']
        else:
            self.logger.error('db config not provided')
            return False

        self.model_id = self.config.get('model_id', '')
        if not self.model_id:
            self.logger.error('model id not provided in config')
            return False
        self.input_annotation_file = self.config.get('annotation_file', '')
        if not self.check_path_exists(self.input_annotation_file):
            return False

        #check output file name
        self.train_filename = self.module_config['train'].get('train_file', '')
        if not self.train_filename:
            self.logger.error('train output file name empty')
            return False
        self.validate_filename = self.module_config['train'].get('validate_file', '')
        if not self.validate_filename:
            self.logger.error('validate output file name empty')
            return False
        self.test_filename = self.module_config['train'].get('test_file', '')
        if not self.test_filename:
            self.logger.error('test output file name empty')
            return False
        self.anno_filename = self.module_config['train'].get('annotation_file', '')
        if not self.anno_filename:
            self.logger.error('annotation output file name empty')
            return False

        self.logger.info('checking other output parameters')
        #load output distribution pcts
        self.train_pct = self.config.get('train_pct', 0.0)
        self.validate_pct = self.config.get('validate_pct', 0.0)
        self.test_pct = self.config.get('test_pct', 0.0)
        self.probabilities = [('train', self.train_pct), ('validate', self.train_pct+self.validate_pct), ('test', self.train_pct+self.validate_pct+self.test_pct)]

        #ctgy##Added for bypassing the patch extraction
        self.ctgy = self.config.get('ctgy', 'rad')
        self.colorspace = self.config.get('colorspace', 'gray')
        if self.ctgy == 'rad':
            self.img_params=self.config.get('img_params',[0,0,256,256])     #used in rad ctgy for inputreader    
                                  #TODO: Ask for schema change here. Different categories shoudl have different pipelines    
        self.out_ppm = self.config.get('ppm', [])
        if not self.out_ppm:
            self.logger.error('output ppm value not provided')
            return False

        self.patch_size = self.config.get('output_size_um', 0)
        if not self.patch_size:
            self.logger.error('output patch size not provided')
            return False
        #convert patch_size to pixels
        self.patch_size = [int(self.patch_size*self.out_ppm[0]), int(self.patch_size*self.out_ppm[1])]

        #output delimiter
        self.output_delim = self.config.get('output_delim', '|')

        #save output?
        self.save_images = self.config.get('save_images',False)

        #load post processing operations
        self.logger.info('loading post processing operations')
        self.image_procs = []
        for ppname in self.config.get('image_procs', []):
            self.image_procs.append(baseimageproc.factory(ppname, self.logger, self.config))

        #load variants
        self.logger.info('loading variant image operations')
        self.variants = {}
        for variant in self.config.get('variants', []):
            for ds in ['train', 'validate', 'test']:
                self.variants[(variant, ds)] = baseimagevariant.factory(variant, self.logger, self.config, ds)

        #histogram matching
        self.histogram_matching = ('histogrammatch' in self.config.get('image_procs', []))
        self.reference_histogram = self.config.get('reference_histogram', [])
        if self.histogram_matching and not self.reference_histogram:
            raise Exception('no reference histogram given for histogram matching')

        #global lcn
        self.global_lcn = ('globallcn' in self.config.get('image_procs', []))

        #region model params
        self.region_patch_size = self.config.get('region_patch_size', 16)
        self.inverted_planes = ('invertplanes' in self.config.get('image_procs', []))

        #background removal
        self.remove_background_cell = self.config.get('remove_background_cell', '')

        #shuffle
        self.shuffle = self.config.get('shuffle', False)
       
        self.logger.info('config loaded and verified')
        return True

    def get_input_path(self, input_id):
        if self.dbclient:
            self.logger.info('fetching input path for %s from DB' % input_id)
            (mdocs, status) = self.dbclient.get_data(self.db, self.input_control_table, {'file_name': input_id})
            #check if input file is usable or is awaiting sync from remote instance
            input_status = mdocs[0].get('status', '')
            sync_status = mdocs[0].get('sync_status', '')
            instance_id = mdocs[0].get('instance_id', None)
            if input_status != "success" or ( sync_status not in ['sync_pending:%s' % self.sys_config.get('instance_id', ''), "success"] and instance_id != self.sys_config.get('instance_id','')):
                raise Exception(unicode('cannot use input file %s%s with status %s and sync_status %s' % (mdocs[0]['file_name'], mdocs[0]['file_ext'], input_status, sync_status)).encode('utf-8'))
            return (mdocs[0]['file_type'], os.path.join(mdocs[0]['file_dest'], '%s%s' % (mdocs[0]['file_name'], mdocs[0]['file_ext'])), [mdocs[0]['ppm_x'],mdocs[0]['ppm_y']], mdocs[0].get('set_id', None))
        else:
            return (None, None, None, None)

    def check_path_exists(self, path):
        if not os.path.exists(path):
            self.logger.info('Path %s does not exist' % (path))
            return False
        self.logger.info('Path %s exists' % (path))
        return True

    def shuffle_files(self):
        if not self.shuffle or not self.shuffle_keys:
            return
        self.logger.info('shuffling files')
        for key in self.shuffle_keys:
            if key not in self.outfps:
                self.logger.info('ignoring type %s - files not present in output file list' % key)
                continue
            self.logger.info('shuffling for key %s' % key)
            #close file pointer
            file_path = self.outfps[key].name
            self.outfps[key].close()
            
            #shuffle
            file_info = os.stat(file_path)
            file_size = file_info.st_size

            #assumption: atleast 1GB RAM will be available for the shuffle
            num_buckets = file_size/(1024*1024*1024)+1

            #split file into buckets
            bucket_fps = []
            for i in xrange(num_buckets):
                bucket_fps.append(DataFile('%s.b%d' % (file_path, i), 'w', self.logger).get_fp())
            self.outfps[key] = DataFile(file_path, 'r', self.logger).get_fp()
            bindex = 0
            for line in self.outfps[key]:
                bucket_fps[bindex].write(line)
                bindex = (bindex + 1) % num_buckets

            #open file pointer again
            self.outfps[key] = DataFile(file_path, 'w', self.logger).get_fp()
            for i in xrange(num_buckets):
                bpath = bucket_fps[i].name
                bucket_fps[i].close()
                bfp = DataFile(bpath, 'r', self.logger).get_fp()
                lines = bfp.readlines()
                bfp.close()
                random.shuffle(lines)
                for line in lines:
                    self.outfps[key].write(line)

            #cleanup bucket files
            for i in xrange(num_buckets):
                bpath = '%s.b%d' % (file_path, i)
                if os.path.exists(bpath):
                    os.remove(bpath)

            self.logger.info('shuffling for key %s complete' % key)

    def create_training_data(self):
        self.logger.info('loading label map')
        #get model labels from db
        self.logger.info('fetching labels for model %s from db' % self.model_id)
        (recs, status) = self.dbclient.get_data(self.db, self.model_table, {'id': self.model_id})
        #load model info
        self.label_map = {}
        cnt = 0
        for label in recs[0]['labels']:
            self.label_map[label] = float(cnt)
            cnt += 1
        self.model_type = recs[0]['type']
        self.negative_label = recs[0].get('negative_label', None)
        if self.model_type == 'region' and len(self.variants) > 0:
            raise Exception('image variants not supported for region models')

        #create db entry for run
        if "train_data" in self.sys_config['sync_entities']:
            sync_status = 'sync_pending:%s' % self.sys_config.get('instance_id', '')
        else:
            sync_status = "not_synced"
        (self.tdid, status) = self.dbclient.post_data(self.db, self.train_data_table, {'instance_id': self.sys_config.get('instance_id', ''), 'path':'', 'status':'in process', 'model_id': self.model_id, 'ts': datetime.utcnow(), 'config': self.config, 'sync_status': sync_status})
        if status:
            raise Exception('could not create entry in train_data_control table')
        self.logger.info('db entry created for current run')

        try:
            self.output_dir = os.path.join(self.train_data_base_path, self.model_id, datetime.utcnow().strftime('%Y%m%d'), str(self.tdid))
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            self.logger.info('creating output files in %s' % (self.output_dir))

            #if the save_image is true, then write all the images into the image
            #directory. The below code is to get that done!
            self.logger.debug('save_images value = %s' % str(self.save_images))
            if self.save_images:
                #create the folder for images, if required
                self.image_dir = os.path.join(self.output_dir, 'images')
                if not os.path.exists(self.image_dir):
                    os.makedirs(self.image_dir)
                    self.logger.info('creating directory to store image files in %s' % (self.image_dir))
            #open outfile pts
            self.logger.info('opening output files for writing')
            self.outfps['train'] = DataFile(os.path.join(self.output_dir, self.train_filename), 'w', self.logger).get_fp()
            self.outfps['validate'] = DataFile(os.path.join(self.output_dir, self.validate_filename), 'w', self.logger).get_fp()
            self.outfps['test'] = DataFile(os.path.join(self.output_dir, self.test_filename), 'w', self.logger).get_fp()
            self.outfps['anno'] = DataFile(os.path.join(self.output_dir, self.anno_filename), 'w', self.logger).get_fp()

            self.outcnts = {}
            self.input_path_info = {}
            self.input_params_info = {}
            self.bgd_removal_info = OrderedDict()
           
            self.cell_extractor = None
            if self.remove_background_cell:
                self.cell_extractor = cell.factory(self.remove_background_cell, self.module_config['dataextraction'].get(self.remove_background_cell, {}), self.remove_background_cell, True, self.out_ppm, self.logger)
                self.cell_stride_size = self.module_config['dataextraction'].get(self.remove_background_cell, {}).get('stride_size', 1024)

            #pre processing for histogram matching and/or global lcn
            if self.histogram_matching or self.global_lcn:
                self.imageparamutils = ImageParamUtils(self.sys_config, self.logger)
                if self.histogram_matching:
                    #calculate average cumulative histogram of all input sets 
                    self.logger.info('calculating cumulative histogram of input sets')
                if self.global_lcn:
                    self.logger.info('calculating global lcn of input files')
                
                af = DataFile(self.input_annotation_file, 'r', self.logger, sort=True).get_fp()
                for line in af:
                    patch_id, label = line.strip().split(',', 1)
                    patch_id, patch_ext = patch_id.split('.', 1)
                    patch_fields = patch_id.split('_')
                    input_id = patch_fields[0]
                    (input_type, input_path, input_ppm, input_set_id) = (None, None, None, None)
                    if input_id not in self.input_path_info:
                        (input_type, input_path, input_ppm, input_set_id) = self.get_input_path(input_id)
                        self.input_path_info[input_id] = (input_type, input_path, input_ppm, input_set_id)
                    else:
                        (input_type, input_path, input_ppm, input_set_id) = self.input_path_info[input_id]
                    #if input_set_id not in self.input_params_info:
                    if input_id not in self.input_params_info:
                        if self.histogram_matching:
                            self.logger.info('calculating histogram for %s' % input_id)
                            self.input_params_info.setdefault(input_id, {})['hist'] = self.imageparamutils.get_input_file_histogram(input_id, input_set_id)
                        if self.global_lcn:
                            self.logger.info('calculating global lcn for %s' % input_id)
                            self.input_params_info.setdefault(input_id, {})['lcn'] = self.imageparamutils.get_input_file_lcn(input_id, input_set_id)
                af.close()
                if self.histogram_matching:
                    #calculate mapping of all input sets with reference histogram
                    self.logger.info('calculating histogram mapping of input sets')
                    for set_id, set_info in self.input_params_info.iteritems():
                        self.logger.info('calculating histogram mapping for %s' % set_id)
                        set_info['histmatch_pixmap'] = self.imageparamutils.get_histogram_mapping(self.reference_histogram, set_info['hist'])
            #iterate over annotation file
            self.logger.info('processing annotation file')
            af = DataFile(self.input_annotation_file, 'r', self.logger).get_fp()
            for line in af:
                self.outfps['anno'].write(line)
                self.outcnts['anno'] = self.outcnts.get('anno', 0)+1
                patch_id, label = line.strip().split(',', 1)
                patch_id, patch_ext = patch_id.split('.', 1)
                patch_fields = patch_id.split('_')
                self.logger.debug('processing patch %s with label %s' % (patch_id, label))

                input_id = patch_fields[0]
                if self.ctgy == 'rad':              #hacked for avoiding the img_name grammar change for testing. ##FIX
                    x=0
                    y=0
                else:
                    x = int(patch_fields[2])
                    y = int(patch_fields[3])                        

                regions = []
                if self.model_type == 'region':
                    for linfo in label.split(';'):
                        lname,pts = linfo.split(':', 1)
                        pts = [[int(self.patch_size[0]*float(a)),int(self.patch_size[1]*float(b))] for (a,b) in [z.split('|', 1) for z in pts.split('#')]]
                        regions.append((lname,numpy.array(pts)))

                input_reader = self.input_readers.get(input_id, None)
                (input_type, input_path, input_ppm, input_set_id) = self.input_path_info.get(input_id, (None, None, None, None))
                #init input reader if not already done
                if not input_reader:
                    if input_id in self.input_path_info:
                        (input_type, input_path, input_ppm, input_set_id) = self.input_path_info.get(input_id)
                    else:
                        (input_type, input_path, input_ppm, input_set_id) = self.get_input_path(input_id)
                        self.input_path_info[input_id] = (input_type, input_path, input_ppm, input_set_id)
                    self.logger.info('opening %s input file %s from %s' % (input_type, input_id, input_path))
                    input_reader = baseinputreader.factory(input_type, self.logger, input_path, in_ppm=input_ppm, out_ppm=self.out_ppm)
                    #check if max open readers crossed
                    if len(self.input_readers) >= self.max_open_readers:
                        (rid, rval) = self.input_readers.popitem(last=False)
                        rval.close()
                        self.logger.info('max open readers limit (%d) reached, closing oldest reader %s' % (self.max_open_readers, rid))
                    self.input_readers[input_id] = input_reader

                self.logger.debug('using input %s' % (input_id))

                #get a bigger patch if label model 
                self.logger.debug('reading patch from input with centroid at (%d,%d)' % (x, y))
                pimg = None
                if self.ctgy == 'rad':
                    pimg = input_reader.get_patch(self.img_params[0],self.img_params[1],self.img_params[2], self.img_params[3]) 
                    # if self.colorspace == 'gray':
                    #     if pimg.shape[-1] == 3: 
                    #         pimg = cv2.cvtColor(pimg, cv2.COLOR_BGR2GRAY)
                                
                else:
                    if self.model_type == 'region':
                        pimg = input_reader.get_patch(x-self.patch_size[0]/2, y-self.patch_size[1]/2, self.patch_size[0], self.patch_size[1])
                    else:
                        pimg = input_reader.get_patch(x-self.patch_size[0], y-self.patch_size[1], 2*self.patch_size[0], 2*self.patch_size[1])
                
                            

                #apply background removal if applicable
                if self.remove_background_cell:
                    rembgd_key = '%s_%d_%d' % (input_id, (x-x%self.cell_stride_size), (y-y%self.cell_stride_size))
                    remove_bgd_info = self.bgd_removal_info.get(rembgd_key, None)
                    if None is remove_bgd_info:
                        self.logger.info('calculating background removal attributes for key %s' % rembgd_key)
                        if len(self.bgd_removal_info) >= self.max_open_readers:
                            (k,v) = self.bgd_removal_info.popitem(last=False)
                            self.logger.info('max cached background removal info reached, closing oldest item %s' % k)
                            del v
                            del k
                        remove_bgd_info = self.bgd_removal_info.setdefault(rembgd_key, self.cell_extractor.get_background_removal_info(input_reader, x, y))
                    if None is remove_bgd_info or not remove_bgd_info:
                        raise Exception('could not get background removal info for cell type %s, input %s' % (self.remove_background_cell, input_id))
                    if not self.cell_extractor.remove_background(pimg, remove_bgd_info[0], remove_bgd_info[1], x, y):
                        self.logger.error('could not remove background for patch %s, ignoring it' % (patch_id))
                        continue

                #decide which set the image will end up in
                rnd = random.random()
                dataset = None
                for (ds, prob) in self.probabilities:
                    if rnd < prob:
                        dataset = ds
                        break
                if not dataset:
                    self.logger.error('image %s could not be assigned any dataset from train, validate, test; prob %f; probarray %s' % (patch_id, rnd, str(self.probabilities)))
                    continue
                self.logger.debug('patch random number: %f, chosen dataset: %s' % (rnd, dataset))

                #if the save_image is true, then write all the images into the image
                #directory. The below code is to get that done!
                if self.save_images:
                    self.logger.debug('save_images is set to true. saving the image to '+self.image_dir)
                    try:
                        cv2.imwrite(os.path.join(self.image_dir,patch_id+".jpg").encode("utf-8"),pimg)
                        #Image.fromarray(pimg).save(os.path.join(self.image_dir,patch_id) + ".jpg","JPEG")
                    except Exception as e:
                        self.logger.exception('exception caught while writing image to disk')
                else:
                    self.logger.debug('save_images is set to false. skipping saving output')

                #create list of regions sorted by size ascending
                if self.model_type == 'region':
                    nregions = []
                    regions.sort(key=lambda x: cv2.contourArea(x[1]))
                    for (lname,pts) in regions:
                        #create a region mask
                        mask = numpy.zeros([pimg.shape[0],pimg.shape[1],1], numpy.uint8)
                        cv2.drawContours(mask, [pts], 0, 255, -1)
                        nregions.append((lname, mask))
                    regions = nregions

                #generate variants
                self.logger.debug('generating variants')
                img_variants = []
                #add current image to start off with
                img_variants.append(('',pimg))
                #add all configured variants
                for (variant,ds) in self.variants.iterkeys():
                    if ds != dataset:
                        continue
                    img_variants.extend(self.variants[(variant,ds)].get_variants(patch_id, pimg, label))

                #post processing
                self.logger.debug('post processing %d variants' % (len(img_variants)))
                for (key,img) in img_variants:
                    extra_fields = []
                    for image_proc in self.image_procs:
                        (img, info) = image_proc.process_img(patch_id, img, self.input_params_info.get(input_id, {}))
                        extra_fields.extend(info)
                    #write image to file
                    if self.model_type == 'region':
                        #create patches of image and mark label as per region
                        stepx,stepy = self.region_patch_size/2,self.region_patch_size/2
                        i,j = 0,0
                        while i+self.region_patch_size < self.patch_size[0]:
                            while j+self.region_patch_size < self.patch_size[1]:
                                centx = i+self.region_patch_size/2-1
                                centy = j+self.region_patch_size/2-1
                                match_label = self.negative_label
                                for (lname,mask) in regions:
                                    if numpy.sum(mask[centy:centy+2,centx:centx+2])/255. >= 3.0:
                                        match_label = lname
                                        break
                                patch_img = None
                                if self.inverted_planes:
                                    patch_img = img[:,j:j+self.region_patch_size,i:i+self.region_patch_size]
                                else:
                                    patch_img = img[j:j+self.region_patch_size,i:i+self.region_patch_size,:]
                                out_fields = []
                                out_fields.append('%s%s.%s' % (patch_id, '_%s' % (key) if key else '', patch_ext))
                                out_fields.append(base64.b64encode(numpy.ascontiguousarray(patch_img, dtype=numpy.float32)))
                                out_fields.append(base64.b64encode(numpy.array(self.label_map[match_label], dtype=numpy.float32)))
                                out_fields.extend(extra_fields)
                                self.logger.debug('writing %s to %s dataset after processing' % (out_fields[0], dataset))
                                self.outfps[dataset].write('%s\n' % self.output_delim.join(out_fields))
                                self.outcnts[dataset] = self.outcnts.get(dataset, 0)+1
                                j += stepy
                            j = 0
                            i += stepx
                    else:
                        out_fields = []
                        out_fields.append('%s%s.%s' % (patch_id, '_%s' % (key) if key else '', patch_ext))
                        out_fields.append(base64.b64encode(numpy.ascontiguousarray(img, dtype=numpy.float32)))
                        out_fields.append(base64.b64encode(numpy.array(self.label_map[label], dtype=numpy.float32)))
                        out_fields.extend(extra_fields)
                        self.logger.debug('writing %s to %s dataset after processing' % (out_fields[0], dataset))
                        self.outfps[dataset].write('%s\n' % self.output_delim.join(out_fields))
                        self.outcnts[dataset] = self.outcnts.get(dataset, 0)+1
            af.close()
            self.logger.info('processed annotation file')

            #shuffle files if required
            self.shuffle_files()

            #update db status and path
            self.logger.info('updating db with information')
            (num_updated, status) = self.dbclient.update_data(self.db, self.train_data_table, {'_id': self.tdid}, {'$set': {'path': self.output_dir, 'status':'success', 'ts': datetime.utcnow(), 'counts': self.outcnts}})
            if status:
                raise Exception('could not update success status for db row %s in train_data_control' % str(self.tdid))
            #schedule for sync
            if "train_data" in self.sys_config['sync_entities']:
                self.logger.info('scheduling training data for sync')
                sync_record = {'instance_id': self.sys_config.get('instance_id', ''), 'sync_type': 'train_data', 'sync_paths': [os.path.join(self.output_dir, self.anno_filename)], 'status': 'pending', 'source_table': self.train_data_table, 'source_key': str(self.tdid), 'ts': datetime.utcnow()}
                (syncid, status) = self.dbclient.post_data(self.db, self.sync_table, sync_record)
                if status:
                    raise Exception('could not schedule sync of training data id %s' % str(self.tdid))
                self.logger.info('sync for training data %s scheduled successfully' % str(self.tdid))
            return self.tdid
        except Exception as e:
            #update failure status in db
            self.logger.exception('caught exception, updating failure status in db')
            (num_updated, status) = self.dbclient.update_data(self.db, self.train_data_table, {'_id': self.tdid}, {'$set': {'status': 'failure', 'ts': datetime.utcnow()}})
            if status:
                self.logger.error('could not update failure status for db row %s in train_data_control' % str(self.tdid))
            raise

    def cleanup(self):
        self.logger.info('cleaning up TrainingDataCreator')
        #close outfps
        for key in self.outfps.iterkeys():
            self.logger.info('closing %s file pointer' % key)
            self.outfps[key].close()
        #close input_readers
        for key in self.input_readers.iterkeys():
            self.logger.info('closing %s input pointer' % key)
            self.input_readers[key].close()
        #TODO: close db client
