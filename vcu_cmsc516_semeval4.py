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

PATTERN_FOR_DOC_ID = '(?:\/.*-s)([0-9]*)(?:[a-z])([0-9]*)'

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

# model name this is the name of the model to create with new map or add to or to use with an evaluation
pars.add_argument('-m', '--model',
                  help='Model to be used/created/extended by the application this item should be specified always.',
                  dest='model',
                  required=True)

# map file this is the entity map file for the named entities
pars.add_argument('-mf', '--mapFile',
                  help='name of entity map the file that contains the entity names and their entity IDs',
                  type=file,
                  dest='map_file')

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
# arguments passed in the command line. This will need to be broken out to deal with all the disjoint
# sets of operations we want:
#     1: train and pickle training data
#     2: init the model
#     3: evaluate from the current model
arguments = pars.parse_args()
d = vars(arguments)

# first thing load the model if there is one if there's not a model a new model will be built
# with the model passed in to the applicaiton
sEcf.load_model(d['model'])

# if a map file is supplied load the entity map and add it to the model
if d['map_file'] is not None:
    # set the map and close the file because we don't need it anymore
    sEcm.entity_map = sEcf.build_entity_dict(d['map_file'])
    d['map_file'].close()
    # update the model
    sEcf.update_model(sEcm.MODEL_ENTITY_MAP, sEcm.entity_map)

# columns object will always exist because there is a default list of columns so we can set the columns
# the training data uses the Default_headings in the model python file so we dont need to care about dealing
# with in in a specific way
if d['columns'] != sEcm.DEFAULT_HEADINGS:
    sEcm.DEFAULT_HEADINGS = d['columns']

# we can now get the words, speakers and the entities they are referencing using the probability function
# we will store these dictionaries as part of our model.
# TODO make sure we can update the model with new training data and keep the existing trained info
entity_mentions_and_counts = None
if d['train_file'] is not None:
    entity_mentions_and_counts, words, speakers = sEcf.build_basic_probability_matrix(d['train_file'])
    d['train_file'].close()
    sEcf.update_model(sEcm.MODEL_DISTRIBUTIONS, entity_mentions_and_counts)
    sEcf.update_model(sEcm.MODEL_WORDS, words)
    sEcf.update_model(sEcm.MODEL_SPEAKERS, speakers)

sEcf.save_model(d['model'])

# gracefully end the application
# TODO come up with something more meaning ful
print('finished: there are {} speakers'.format(len(entity_mentions_and_counts)))
