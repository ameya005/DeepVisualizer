#!/usr/bin/env python

''' logger.py: Module to initialize and log code'''
__author__ = "Rohit K. Pandey"
__copyright__ = "Copyright 2015, SigTuple Technologies Pvt. Ltd"
__version__ = "0.1"
__email__ = "rohit@sigtuple.com"


import logging
from logging.handlers import TimedRotatingFileHandler
class Logger(object):

    def __init__(self,name,path,debug_flag,console_log,rotate_flag=False):
        level_def = { 'debug':logging.DEBUG,
                    'info':logging.INFO,
                    'warning':logging.WARNING,
                    'error':logging.ERROR,
                    'critical':logging.CRITICAL,
                    }

        level = level_def.get(debug_flag,logging.NOTSET)

        logging.basicConfig(level=level)
        logger = logging.getLogger(name)
        if rotate_flag:
            hdlr = TimedRotatingFileHandler(path,when='H',interval=1,backupCount=168) ##log rotation for 1 (H)our and backing up max of 24*7 (=168) files.
            hdlr.suffix = "%Y%m%dT%H"
        else:
            hdlr = logging.FileHandler(path)
        formatter = logging.Formatter(u'%(asctime)s|%(levelname)s|%(filename)s|%(funcName)s|%(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)

        if console_log:
            logger.propagate = True
        else:
            logger.propagate = False

        self.logger = logger

    def get_logger(self):
        return self.logger

