#!/usr/bin/env python

import os,sys
import json
import re
import codecs
import numpy
import cv2
import math
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from basereportgenerator import BaseReportGenerator
from modules.utils.fsutils import DataFile

class LatexReportGenerator(BaseReportGenerator):

    def __init__(self, sys_config, analysis_config, logger):
        self.sys_config = sys_config
        self.analysis_config = analysis_config
        self.logger = logger
        self.logger.info('latex report generator initialised successfully')


    def get_patch_path(self, keys, data):
        #create image of max 100 patches
        patches = data[:100]

        #border size
        border_size = 2
        num_width = 5
        batch_size = 5
        image_size = 64
        if patches:
            timg = cv2.imread(patches[0]['name'].encode('utf-8'))
            if timg.shape[0] <= 64:
                num_width=10
            image_size = timg.shape[0]
            del timg
        canvas_width = (image_size+border_size)*num_width+border_size
        canvas_height = int(math.ceil(len(patches)/(num_width*1.)))*(image_size+border_size)+border_size
        #create canvas
        canvas = numpy.zeros((canvas_height, canvas_width, 3), dtype=numpy.uint8)
        for i in xrange(len(patches)):
            img = cv2.imread(patches[i]['name'].encode('utf-8'))
            y = border_size+(border_size+image_size)*(i/num_width)
            x = border_size+(border_size+image_size)*(i%num_width)
            canvas[y:y+image_size,x:x+image_size] = img
        #write out canvas in batches of 5
        outfiles = []
        for i in xrange(int(math.ceil(((canvas_height-border_size)/(image_size+border_size))/(1.0*batch_size)))):
            outfile = os.path.join(self.output_dir, '%s_%d.png' % ('_'.join(keys), i))
            cv2.imwrite(outfile.encode('utf-8'), canvas[i*(batch_size)*(image_size+border_size):min(canvas_height, border_size+(i+1)*(batch_size)*(image_size+border_size))])
            outfiles.append(outfile)
        return '\n'.join(['\includegraphics[width=\\textwidth]{%s}' % x for x in outfiles])

    def get_histogram_path(self, keys, data, global_data=None):
        num_bins = len(data)
        xlabels = []
        yvals = []
        y1vals = []
        for i, val in enumerate(data):
            xlabels.append(val[0])
            yvals.append(val[1])
            if None is not global_data:
                y1vals.append(global_data[i][1])
        width=1.0
        pos = numpy.arange(num_bins)
        ax = pyplot.axes()
        ax.set_xticks(pos + (width/2))
        ax.set_xticklabels(xlabels, rotation=45)

        pyplot.bar(pos, yvals, width, color='r')
        if None is not global_data:
            pyplot.plot(pos+width/2, y1vals, marker='o')
        pyplot.xlim(pos.min(), pos.max()+width)
        outfile = os.path.join(self.output_dir, '%s.png' % ('_'.join(keys)))
        pyplot.tight_layout()
        pyplot.savefig(outfile)
        pyplot.close()
        return outfile

    def get_scatter_path(self, keys, data):
        axes = data.get('axes', [])
        vals = numpy.array(data.get('data', []))
        scatter_range = data.get('range', [0,100])
        outfile = ''

        if None is axes or len(axes) < 2:
            self.logger.error('too few axes to create scatter plot')
            return outfile

        if len(axes) >= 3:
            fig = pyplot.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(vals[:,0], vals[:,1], vals[:,2], c='r', marker='o')
            ax.set_xlabel(axes[0])
            ax.set_ylabel(axes[1])
            ax.set_zlabel(axes[2])
            ax.set_xlim(scatter_range)
            ax.set_ylim(scatter_range)
            ax.set_zlim(scatter_range)
            ax.set_autoscale_on(False)
        elif len(axes) >= 2:
            fig = pyplot.figure()
            ax = fig.add_subplot(111)
            ax.scatter(vals[:,0], vals[:,1], c='r', marker='o')
            ax.set_xlabel(axes[0])
            ax.set_ylabel(axes[1])
            ax.set_xlim(scatter_range)
            ax.set_ylim(scatter_range)
            ax.set_autoscale_on(False)

        outfile = os.path.join(self.output_dir, '%s.png' % ('_'.join(keys)))
        pyplot.tight_layout()
        pyplot.savefig(outfile)
        pyplot.close()
        return outfile

    def get_replace_str(self, template_str, data):
        keys = template_str.split(':')
        keydata = data
        keydataparent = data
        for i in xrange(len(keys)):
            key = keys[i]
            keydataparent = keydata
            if i == len(keys)-1:
                keydata = keydata.get(key, '')
            else:
                keydata = keydata.get(key, {})
        if 'histogram' in keys:
            if type(keydata) is dict:
                return self.get_histogram_path(keys, keydata.get('hist', []), keydata.get('global', None))
        elif 'scatter' in keys:
            if type(keydata) is dict:
                return self.get_scatter_path(keys, keydata)
        elif 'patch' in keys:
            if type(keydata) is list:
                return self.get_patch_path(keys, keydata)
        return str(keydata)

    def generate_report(self, data_json_path, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        #load data json
        inf = DataFile(data_json_path, 'r', self.logger).get_fp()
        self.data = json.load(inf)
        inf.close()

        #load report template
        template_path = os.path.join(self.sys_config['paths']['template'], '%s.%s.template' % (self.analysis_config['analyser']['id'], 'latex'))
        inf = DataFile(template_path, 'r', self.logger).get_fp()
        template = inf.read()
        inf.close()

        #start replacing strings
        template_regex = re.compile(r'###(?P<key>[^#]*)###')
        replaced = template
        for match in template_regex.finditer(template):
            replaced = replaced.replace(match.group(0), self.get_replace_str(match.group('key'), self.data))
        out_path = os.path.join(self.output_dir, 'output.tex')
        outf = DataFile(out_path, 'w', self.logger).get_fp()
        outf.write(replaced)
        outf.close()
        #link to images directory
        cmd = u'cd %s; ln -s %s .' % (self.output_dir, os.path.join(self.sys_config['paths']['template'], 'images'))
        os.system(cmd.encode('utf-8'))
        #create pdf
        cmd = 'cd %s; pdflatex -interaction nonstopmode output.tex output.pdf 1>>/dev/null' % self.output_dir
        os.system(cmd.encode('utf-8'))
        out_path = os.path.join(self.output_dir, 'output.pdf')
        return out_path

    def cleanup(self):
        pass
