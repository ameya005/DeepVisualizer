import os
import numpy
import cv2
import glob
import json
from scipy.spatial import distance
from modules.utils.fsutils import DataFile
from basemodelpostproc import BaseModelPostProc

class RegionAttributes(BaseModelPostProc): 

    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.region_label = self.config.get('region_label', None)
        self.ppm = self.config.get('ppm', [1.0,1.0])

        def pixel_thresh(thresh, a):
            if a != thresh:
                return 0
            else:
                return 255

        self.pixel_thresh_func = numpy.vectorize(pixel_thresh, otypes=[numpy.uint8])

    def process_model_output(self, model_labels, model_out_dir):
        #find index of label
        label_index = model_labels.index(self.region_label)
        pixel_val = int((255.*label_index)/max(1, len(model_labels)-1))

        #get list of images to process
        files = glob.glob(os.path.join(model_out_dir, '*.*'))
        self.logger.info('found %d files to process' % len(files))

        #create output file with information
        self.output_file = os.path.join(model_out_dir, 'region_attributes')
        outf = DataFile(self.output_file, 'w', self.logger).get_fp()
        for fpath in files:
            img = None
            try:
                img = cv2.imread(fpath.encode('utf-8'))
            except:
                img = None

            if None is img or not img.any():
                self.logger.info('could not read %s' % fpath)
                continue

            #calculate attributes of region
            fname = os.path.basename(fpath)
            info = {'name': fname}
            
            #remove other regions from image
            img = self.pixel_thresh_func(pixel_val, img)
            img = cv2.threshold(img[:,:,1], 200, 255, cv2.THRESH_BINARY)[1]

            #find contours
            cntrs = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

            #find the contour enclosing the centre of the image
            chosen_cntr = None
            for cntr in cntrs:
                if cv2.pointPolygonTest(cntr, (img.shape[1]/2, img.shape[0]/2), False) >= 0:
                    chosen_cntr = cntr
                    break
            if None is chosen_cntr:
                self.logger.info('could not find a contour enclosing the centre of the image %s' % fname)
                continue

            #find minimum enclosing rectange of contour
            min_rect = cv2.minAreaRect(chosen_cntr)
            box = numpy.int0(cv2.boxPoints(min_rect))
            #find bigger side
            side_lengths = []
            for i in xrange(1, len(box)):
                side_lengths.append(distance.euclidean(box[0], box[i]))
            side_lengths.sort(reverse=True)
            info['dia'] = (side_lengths[1]/self.ppm[0])
            outf.write('%s\n' % json.dumps(info, ensure_ascii=False, encoding='utf-8'))
        outf.close()
        return self.output_file
