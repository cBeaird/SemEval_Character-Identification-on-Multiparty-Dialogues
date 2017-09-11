#!/usr/bin/env python
"""
Application start for the SemEval 2018 task 4 application: conference resolution for SemEval 2018

This file is the main starting point for the application. This files allow for the application to
be called with a series of input parameters to perform the different functions required by the
application. The core functionality is in the semEval_core_functions python file while the core
model data is in the semEval_core_model file additional information can be found in these files.

"""
import argparse
import semEval_core_model as sEcm
import semEval_core_functions as sEcf

__author__ = 'Casey Beaird'
__credits__ = ['Casey Beaird', 'Chase Greco', 'Brandon Watts']
__license__ = 'MIT'
__version__ = '0.1'

# build the command line parser and add the options
pars = argparse.ArgumentParser(usage='Main python file for project package \'VCU CMSC516 SemEval Task4\'',
                               formatter_class=argparse.RawTextHelpFormatter,
                               description='''
                               Main python file for VCU CMSC 516 SemEval 2018 project task 4: 
                               Character Identification on Multiparty Dialogues
                               This is the conference resolution problem, identify reference in corpora given the 
                               speaker.
                               
                               The application expects an entity map and a training file.
                               example: python vcu_cmsc516_semeval4.py -m ./entityMap.txt -t ./trainingData.conll
                               
                               optionally a list of column headings can be supplied that describe your conll data
                               files. The list options are specified here in this help.''',
                               version='0.1')

# map file this is the entity map file for the named entities
pars.add_argument('-mf', '--mapFile',
                  help='name of entity map the file that contains the entity names and their entity IDs',
                  type=file,
                  dest='model_file')

# training data file
pars.add_argument('-tf', '--trainFile',
                  help='input training file for entity identifier',
                  type=file,
                  dest='train_file')

# column headers for the conll file if this is not specified then the headings file is assumed to
# be in the format from the SemEval 2018 Task 4 format that is specified in the core model file.
pars.add_argument('-d', '--headers',
                  help='column type list for the order of columns in you training file',
                  dest='columns',
                  choices=list(sEcm.DEFAULT_HEADINGS),
                  default=sEcm.DEFAULT_HEADINGS,
                  nargs='*')

# parse the command line arguments this will create the namespace class that gives access to the
# arguments passed in the command line.
arguments = pars.parse_args()

# TODO remove test prints
d = vars(arguments)
print(d)

if d['columns'] != sEcm.DEFAULT_HEADINGS:
    sEcm.DEFAULT_HEADINGS = d['columns']

if d['model_file'] is not None:
    sEcm.entity_map = sEcf.build_entity_dict(d['model_file'])
    d['model_file'].close()

if d['train_file'] is not None:
    entity_mentions_and_counts = sEcf.build_basic_probability_matrix(d['train_file'])

print('finished')
