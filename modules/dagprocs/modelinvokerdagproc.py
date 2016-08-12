#!/usr/bin/env python

import os,sys
import json
import os,sys
import numpy
import base64
import json
import time
import pickle
import cv2
import glob
import codecs
from modules.utils.imageparamutils import ImageParamUtils
from modules.mdd import MDD
from modules.imageprocs import baseimageproc
from modules.modelpostprocs import basemodelpostproc
from basedagproc import BaseDAGProc
from modules.utils.fsutils import DataFile

class ModelInvokerDAGProc(BaseDAGProc):

    def __init__(self, sys_config, config, analysis_config, out_queue, proc_configs, logger=None):
        #basic constructor in base class
        super(ModelInvokerDAGProc, self).__init__(sys_config, config, analysis_config, out_queue, proc_configs, logger=logger)

        #validate config
        self.model_id = self.config.get('model_id', None)
        self.model_version = self.config.get('model_version', None)
        self.model_version_info = None
        if not self.model_version and not self.model_id:
            raise Exception('model version and model id not provided in config for DAG proc id %s' % self.procid)
        self.input_paths = self.config.get('input', [])
        # handling case where no extraction has happened
        #if len(self.input_paths) <= 0 or not reduce(lambda x,y: x or y, map(lambda z:os.path.exists(z), self.input_paths)):
        #    raise Exception('no input path provided in config for DAG proc id %s' % self.procid)
        self.output_dir = self.config.get('output', None)
        if not self.output_dir:
            raise Exception('no output path provided in config for DAG proc id %s' % self.procid)
        self.output_dir = os.path.join(self.output_dir, self.procid)
        self.output_path = None

    def get_active_version_for_model(self):
        (self.model_version_info, status) = self.dbclient.get_one(self.db, self.model_version_table, {'model_id': self.model_id, 'status': 'success', 'sync_status': {'$in': ['success', 'sync_pending:%s' % self.sys_config.get('instance_id', '')]}, 'active': True}, sort=[('stats.model_error',1)])
        if status or not self.model_version_info:
            raise Exception('could not find a valid model version to use for model id %s' % self.model_id)
        self.model_version = self.model_version_info.get('version')
        self.logger.info('using version %s for model id %s' % (self.model_version, self.model_id))

    def init(self):
        #fetch model path from db
        self.dbclient = MDD(self.sys_config.get('db', {}), self.logger)
        if not self.dbclient:
            raise Exception('could not open db connection')
        self.db = self.sys_config['db']['database']
        self.model_version_table = self.sys_config['db']['tables']['model_version']
        self.model_table = self.sys_config['db']['tables']['models']
        self.input_control_table = self.sys_config['db']['tables']['input']
        if self.model_id and not self.model_version:
            self.get_active_version_for_model()
        if not self.model_version_info:
            (self.model_version_info, status) = self.dbclient.get_one(self.db, self.model_version_table, {'version': self.model_version})
            if status:
                raise Exception('could not fetch info for model version %s from db' % self.model_version)
        #check mode status
        model_version_status = self.model_version_info.get('status','')
        model_version_sync_status = self.model_version_info.get('sync_status','')
        model_version_instance_id = self.model_version_info.get('instance_id',None)
        if model_version_status != 'success' or ( model_version_sync_status not in ['success', 'sync_pending:%s' % self.sys_config.get('instance_id', '')] and model_version_instance_id != self.sys_config.get('instance_id', '')):
            raise Exception('cannot use model version %s with status %s' % (self.model_version, self.model_version_info.get('status', '')))
        self.model_path = self.model_version_info['path']
        if not self.model_path or not os.path.exists(self.model_path):
            raise Exception('could not find model file for version %s at %s' % (self.model_version, self.model_path.encode('utf-8')))
        #fetch model info
        if not self.model_id:
            self.model_id = self.model_version.split('.')[3]
        (mdoc, status) = self.dbclient.get_one(self.db, self.model_table, {'id': self.model_id})
        if status:
            raise Exception('could not fetch model info from db for model %s' % (self.model_id))
        self.model_labels = mdoc['labels']
        self.model_type = mdoc['type']
        self.negative_label = mdoc.get('negative_label', None)

        self.logger.info('loading input model parameters from %s' % self.model_path)
        mf = DataFile(self.model_path, 'r', self.logger, is_binary=True).get_fp()
        self.model_info = pickle.load(mf)
        mf.close()
        self.model_config = self.model_info.get('config', {})
        self.filter_config = self.config.get('filter', {})

        self.logger.info('model parameters loaded')

        self.logger.info('loading config parameters from model config')
        self.batch_size = self.config.get('batch_size', 100)
        self.input_wd = self.model_config.get('input_wd')
        self.input_ht = self.model_config.get('input_ht')
        self.input_fields = self.model_config.get('input_xy_indexes', [0,1])
        self.input_features = self.model_config.get('input_features', 1)
        self.input_dim = (self.batch_size, self.input_features, self.input_wd, self.input_ht)
        self.filter_dims = self.model_config.get('filter_dims')
        self.mp_dims = self.model_config.get('mp_dims')
        self.mlp_hidden_dims = self.model_config.get('mlp_hidden_dims')
        self.mlp_output_dim = self.model_config.get('mlp_output_dim')
        self.output_dim = self.model_config.get('output_dim')
        self.cnn_type = self.model_config.get('type')
        self.conv_activation = self.model_config.get('conv_activation')
        self.mlp_hidden_activation = self.model_config.get('mlp_hidden_activation')
        self.mlp_out_activation = self.model_config.get('mlp_out_activation')
        self.dropout_rate = self.model_config.get('dropout_rate', 0.)
        self.debug = self.model_config.get('debug', False)

        self.inverted_planes = False
        self.histogram_match = False
        self.global_lcn = False
        self.logger.info('loading preprocs required to process input images')
        self.image_procs = []
        for ppname in self.config.get('image_procs', []):
            if ppname == 'histogrammatch':
                self.histogram_match = True
            elif ppname == 'globallcn':
                self.global_lcn = True
            elif ppname == 'invertplanes':
                self.inverted_planes = True
            self.image_procs.append(baseimageproc.factory(ppname, self.logger, self.config))
        if self.histogram_match:
            self.reference_histogram = self.config.get('reference_histogram', None)
            if None is self.reference_histogram:
                raise Exception('no reference provided for histrogram matching')

        #load region model parameters
        self.centre_region_size = self.config.get('centre_region_size', 2)
        # 
        self.max_input_load = self.config.get('max_input_load', 102400)
        self.max_input_load += self.batch_size - self.max_input_load % self.batch_size

        #load model post procs
        self.model_post_proc = None
        if 'model_post_proc' in self.config:
            self.model_post_proc = basemodelpostproc.factory(self.config['model_post_proc'], self.logger, self.config)


        self.logger.info('model DAG proc id %s initialised successfully' % self.procid)

    def get_input_set_id(self, input_id):
        if not self.dbclient:
            return None
        self.logger.info('fetching input set_id for %s from DB' % input_id)
        (mdocs, status) = self.dbclient.get_data(self.db, self.input_control_table, {'file_name': input_id})
        #check if input file is usable or is awaiting sync from remote instance
        input_status = mdocs[0].get('status', '')
        if input_status != 'success':
            raise Exception(unicode('cannot use input file %s%s with status %s' % (mdocs[0]['file_name'], mdocs[0]['file_ext'], input_status)).encode('utf-8'))
        return mdocs[0].get('set_id', None)

    def run(self):
        import theano
        import theano.tensor as T
        from modules import cnnutils

        try:
            self.init()
            #load whitelist info if any
            self.whitelist, self.blacklist = None, None
            self.patch_labels = {}
            if self.filter_config:
                self.patch_labels = self.read_input()
            self.input_param_info = {}
            self.input_id_info = {}
            if self.histogram_match or self.global_lcn:
                self.imageparamutils = ImageParamUtils(self.sys_config, self.logger)
                if self.histogram_match:
                    #calculate cumulative hists of input sets
                    self.logger.info('calculating cumulative histogram of input sets')
                if self.global_lcn:
                    #calculate lcn params of input sets
                    self.logger.info('calculating lcn params of input sets')
                for indir in self.input_paths:
                    if not os.path.exists(indir):
                        continue
                    for fpath in glob.glob(os.path.join(indir, '*.*')):
                        fname = os.path.basename(fpath)
                        if self.whitelist or self.blacklist:
                            label = self.patch_labels.get(fname, '')
                            if not label:
                                self.logger.debug('skipping %s because filename not found in filter list' % fname)
                                continue
                            if self.whitelist and label not in self.whitelist and 'all' not in self.whitelist:
                                self.logger.debug('skipping %s due to whitelist' % fname)
                                continue
                            elif self.blacklist and (label in self.blacklist or 'all' in self.blacklist):
                                self.logger.debug('skipping %s due to blacklist' % fname)
                                continue
                        img_id, img_ext = os.path.splitext(fname)
                        patch_fields = img_id.split('_')
                        input_id = patch_fields[0]
                        input_set_id = self.input_id_info.setdefault(input_id, self.get_input_set_id(input_id))
                        if input_id not in self.input_param_info:
                            if self.histogram_match:
                                self.logger.info('calculating histogram for %s' % input_set_id)
                                self.input_param_info.setdefault(input_id, {})['hist'] = self.imageparamutils.get_input_file_histogram(input_id, input_set_id)
                            if self.global_lcn:
                                self.logger.info('calculating global lcn for %s' % input_set_id)
                                self.input_param_info.setdefault(input_id, {})['lcn'] = self.imageparamutils.get_input_file_lcn(input_id, input_set_id)
                if self.histogram_match:
                    #calculate histogram maps of input sets
                    self.logger.info('calculating histogram mapping of input sets')
                    for set_id, set_info in self.input_param_info.iteritems():
                        self.logger.info('calculating histogram mapping for %s' % set_id)
                        set_info['histmatch_pixmap'] = self.imageparamutils.get_histogram_mapping(self.reference_histogram, set_info['hist'])

            rng = numpy.random.RandomState()
         
            index = T.lscalar()
            x = T.tensor4('x')
            y = T.matrix('y')
        
            #create cnn
            self.logger.info('creating CNN')
            cnn = cnnutils.CNN(self.cnn_type, rng, x, self.input_dim, self.filter_dims, self.mp_dims, self.mlp_hidden_dims, numpy.prod(self.mlp_output_dim), 
                        self.conv_activation, self.mlp_hidden_activation, self.mlp_out_activation, dropout_rate=self.dropout_rate)
            cnnutils.set_model_params(cnn, self.model_info.get('params', []))
            self.logger.info('CNN created and initialised with input model parameters')

            self.logger.info('initialising theano symbolic variables and functions')
      
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            self.output_path = os.path.join(self.output_dir, 'model.out')
            self.logger.info('opening output file for writing at %s' % self.output_path)
            outf = DataFile(self.output_path, 'w', self.logger).get_fp()
        
            self.logger.info('loading and pre-processing images to feed model')
    
            invoke_x = None
            invoke_model = None
            image_x = None 
            name_x = None
            coord_x = None
            files = []
            for indir in self.input_paths:
                if not os.path.exists(indir):
                    continue
                files.extend(glob.glob(os.path.join(indir, '*.*')))
            self.logger.info('%d input files found' % len(files))
            num_patches = len(files)
            num_buffer_patches = self.batch_size - (num_patches % self.batch_size)
            num_inputs_per_file = 1
            cnt = 0
            
            for f, fpath in enumerate(files):
                fname = os.path.basename(fpath)
                if self.whitelist or self.blacklist:
                    label = self.patch_labels.get(fname, '')
                    if not label:
                        self.logger.debug('skipping %s because filename not found in filter list' % fname)
                        continue
                    if self.whitelist and label not in self.whitelist and 'all' not in self.whitelist:
                        self.logger.debug('skipping %s due to whitelist' % fname)
                        continue
                    elif self.blacklist and (label in self.blacklist or 'all' in self.blacklist):
                        self.logger.debug('skipping %s due to blacklist' % fname)
                        continue
                img_id, img_ext = os.path.splitext(fname)
                patch_fields = img_id.split('_')
                input_id = patch_fields[0]
                img = cv2.imread(fpath.encode('utf-8'))
                for image_proc in self.image_procs:
                    (img, info) = image_proc.process_img(img_id, img, self.input_param_info.get(input_id, {}))
                if None is image_x:
                    self.imgwd, self.imght = img.shape[1], img.shape[0]
                    if self.inverted_planes:
                        self.imgwd, self.imght = img.shape[-1], img.shape[-2]
                    #initialize arrays based on model type and recalculate size if required
                    if self.model_type == 'region':
                        num_patches *= (self.imgwd/self.centre_region_size)*(self.imght/self.centre_region_size)
                        num_buffer_patches = self.batch_size - (num_patches % self.batch_size)
                        num_inputs_per_file = (self.imgwd/self.centre_region_size)*(self.imght/self.centre_region_size)
                    image_x = numpy.zeros((min(self.max_input_load, (num_patches+num_buffer_patches)), self.input_features, self.input_wd, self.input_ht), dtype=numpy.float32)
                    name_x = []
                    coord_x = []
                    invoke_x = theano.shared(image_x[0:self.batch_size], borrow=True)
                    invoke_model = theano.function(inputs=[], outputs=cnn.get_prediction(), givens = {x: invoke_x})

                #add to input buffer
                if self.model_type == 'region':
                    #breakup image into smaller patches
                    i,j = 0,0
                    while i + self.centre_region_size <= self.imgwd:
                        while j + self.centre_region_size <= self.imght:
                            patch = numpy.zeros((self.input_features, self.input_wd, self.input_ht), dtype=numpy.float32)
                            x1,x2 = max(0,i-(self.input_wd-self.centre_region_size)/2), min(self.imgwd, i+(self.centre_region_size+self.input_wd)/2)
                            y1,y2 = max(0,j-(self.input_ht-self.centre_region_size)/2), min(self.imght, j+(self.centre_region_size+self.input_ht)/2)
                            destx = (self.input_wd-self.centre_region_size)/2-(i-x1)
                            desty = (self.input_ht-self.centre_region_size)/2-(j-y1)
                            if self.inverted_planes:
                                patch[:,desty:desty+(y2-y1),destx:destx+(x2-x1)] = img[:,y1:y2,x1:x2]
                            else:
                                patch[desty:desty+(y2-y1),destx:destx+(x2-x1),:] = img[y1:y2,x1:x2,:]
                            name_x.append(fpath)
                            coord_x.append((i,j))
                            image_x[cnt] = patch
                            cnt += 1
                            j += self.centre_region_size
                        j = 0
                        i += self.centre_region_size
                else:
                    name_x.append(fpath)
                    image_x[cnt] = img
                    cnt += 1

                #if buffer is full/near full or its the last file, run model, write output, and empty buffer
                if cnt + num_inputs_per_file >= len(image_x) or (f == len(files)-1 and cnt > 0):
                    self.batch_model_invoke(cnt, image_x, name_x, coord_x, invoke_model, invoke_x, outf)
                    #empty buffers
                    coord_x, name_x = [], []
                    cnt = 0

            if cnt > 0:
                self.batch_model_invoke(cnt, image_x, name_x, coord_x, invoke_model, invoke_x, outf)

            outf.close()

            #invoke post proc if any
            if None is not self.model_post_proc:
                self.output_path = self.model_post_proc.process_model_output(self.model_labels, self.output_dir)

            if self.out_queue:
                self.out_queue.put({self.procid: {'status': 'success', 'output_path': self.output_path}})

            self.cleanup()
            self.logger.info('model execution run finished for proc id %s' % self.procid)
        except Exception as e:
            self.logger.exception('exception in processing of model invoker proc id %s' % self.procid)
            if self.out_queue:
                self.out_queue.put({self.procid: {'status': 'failure'}})

    def batch_model_invoke(self, cnt, image_x, name_x, coord_x, invoke_model, invoke_x, outf):
        if cnt == 0:
            return
        num_patches = cnt
        num_buffer_patches = (self.batch_size - (num_patches % self.batch_size)) % self.batch_size
        while cnt < num_patches+num_buffer_patches:
            cnt += 1
        self.logger.info('number of inputs %d, number of buffer inputs required to complete batch %d' % (num_patches, num_buffer_patches))
    
        num_runs = (num_patches+num_buffer_patches) / self.batch_size
 
        self.logger.info('executing CNN on input')
        i = 0
        curr_name = None
        curr_image = numpy.zeros((self.imght,self.imgwd,1), dtype=numpy.uint8)
 
        while i < num_runs:
            invoke_x.set_value(image_x[i*self.batch_size:(i+1)*self.batch_size])
            y = numpy.reshape(invoke_model(), [self.batch_size]+self.mlp_output_dim)
            indmax = numpy.argmax(y, axis=1)
            for z in xrange(len(y)):
                if i*self.batch_size+z < num_patches:
                    name = name_x[i*self.batch_size+z]
                    if self.model_type == 'region':
                        if None is curr_name or curr_name != name:
                            if None is not curr_name:
                                cv2.imwrite(os.path.join(self.output_dir, os.path.basename(curr_name)).encode('utf-8'), curr_image)
                                curr_image.fill(0)
                            curr_name = name
                        (cx, cy) = coord_x[i*self.batch_size+z]
                        curr_image[cy:cy+self.centre_region_size,cx:cx+self.centre_region_size,0] = int((indmax[z]*255.)/max(1,len(self.model_labels)-1))
                        name = '%s#%d#%d' % (name, cx, cy)
                    outf.write('%s,%s,%f,%s\n' % (name, self.model_labels[indmax[z]], y[z][indmax[z]], '|'.join(['%s:%s' % (self.model_labels[x], str(y[z][x])) for x in xrange(len(y[z]))])))
            i += 1
        if self.model_type == 'region' and None is not curr_name:
            cv2.imwrite(os.path.join(self.output_dir, os.path.basename(curr_name)).encode('utf-8'), curr_image)
        self.logger.info('processed %d inputs' % num_patches)

    def read_input(self):
        self.blacklist = set(self.filter_config.get('blacklist', []))
        self.whitelist = set(self.filter_config.get('whitelist', []))
        #get file path to open
        file_path = None
        if self.filter_config.get('output_from', ''):
            if self.filter_config['output_from'] in self.proc_configs:
                file_path = self.proc_configs.get(self.filter_config['output_from'], None).get('output_path', None)
                self.logger.debug("Reading output from %s for %s" %(file_path, self.procid))
            else:
                self.logger.error('filter dependency %s not present in done procs, skipping' % (self.filter_config['output_from']))
        elif self.filter_config.get('file_path', ''):
            file_path = self.filter_config.get('file_path', '')

        patch_labels = {}
        if file_path and os.path.exists(file_path):
            self.logger.info('loading white/blacklist info from %s' % file_path)
            inf = DataFile(file_path, 'r', self.logger).get_fp()
            for line in inf:
                fields = line.strip().split(self.filter_config.get('sep', ','))
                fpath = fields[self.filter_config['name_index']]
                label = fields[self.filter_config['label_index']]
                fname = os.path.basename(fpath)
                if self.whitelist or self.blacklist:
                    if not label:
                        self.logger.debug('skipping %s because filename not found in filter list' % fname)
                        continue
                    if self.whitelist and label not in self.whitelist and 'all' not in self.whitelist:
                        self.logger.debug('skipping %s due to whitelist' % fname)
                        continue
                    elif self.blacklist and (label in self.blacklist or 'all' in self.blacklist):
                        self.logger.debug('skipping %s due to blacklist' % fname)
                        continue
                    patch_labels[fname] = label

            inf.close()
        return patch_labels

    def cleanup(self):
        pass

    def recompute(self):

        self.logger.debug("Recomputing %s" % self.pid)

        self.init()
        #updated_labels = self.analysis_config.get('updated_labels',{})

        # Keeping region model output same as original in recomputation
        # This kind of model requires extracted images which may have been deleted permanently
        # As recomputation only happens on label changes, making this assumption that label change doesn't affect the output of region model
        if self.model_type == 'region':
            self.output_path = os.path.join(self.output_dir, 'region_attributes')
            return self.output_path

        self.output_path = os.path.join(self.output_dir, 'model.out')


        # # patches from input dag proc
        # patches = self.read_input()
        #
        # # Write updated patches to output
        # inf = DataFile(self.output_path, 'r+', self.logger).get_fp()
        # lines = []
        # for line in inf:
        #
        #     fields = line.strip().split(',')
        #     patch_path = fields[0]
        #     patch_name = os.path.basename(patch_path)
        #     # If patch is output by input dag proc, update it's label if required and write it
        #     if (not patches) or (patch_name in patches):
        #         if patch_name in updated_labels and updated_labels[patch_name].get('new_label',None) in self.model_labels:
        #             self.logger.info("Changing label for patch:%s from %s to %s in model %s" %(patch_name, fields[1],updated_labels[patch_name]['new_label'],self.model_id))
        #             lines.append('%s,%s,%f,%s\n' %(patch_path, updated_labels[patch_name]['new_label'],1,fields[3]))
        #         else:
        #             lines.append(line)
        # inf.seek(0)
        # inf.writelines(lines)
        # inf.truncate()
        # inf.close()

        return self.output_path
