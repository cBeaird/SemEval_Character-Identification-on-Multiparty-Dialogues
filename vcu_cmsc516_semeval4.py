#!/usr/bin/env python
"""
Core elements of the model for conference resolution for SemEval 2018

This module will contain the dict objects for reference through out the
application. Any constant/format/list that is needed across files should
be housed here. No functions will be defined here as this will be the root
file for the model.

The standard Conll format is used for all input data where the default form is:
# Document ID: /<name of the show>-<season ID><episode ID> (e.g., /friends-s01e01).
# Scene ID: the ID of the scene within the episode.
# Token ID: the ID of the token within the sentence.
# Word form: the tokenized word.
# Part-of-speech tag: the part-of-speech tag of the word (auto generated).
# Constituency tag: the Penn Treebank style constituency tag (auto generated).
# Lemma: the lemma of the word (auto generated).
# Frameset ID: not provided (always "_").
# Word sense: not provided (always "_").
# Speaker: the speaker of this sentence.
# Named entity tag: the named entity tag of the word (auto generated).
# Entity ID: the entity ID of the mention, that is consistent across all documents.

"""
import argparse
import semEval_core_model as scm

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
                               example: python vcu_cmsc516_semeval4.py -m ./entityMap.txt -t ./trainingData.conll''',
                               version='0.1')

pars.add_argument('-m', '--map',
                  help='name of entity map the file that contains the entity names and their entity IDs',
                  type=file,
                  dest='model_file')

pars.add_argument('-t', '--train',
                  help='input training file for entity identifier',
                  type=file,
                  dest='train_file')

pars.add_argument('-d', '--headers',
                  help='column type list for the order of columns in you training file',
                  dest='columns',
                  choices=list(scm.DEFAULT_HEADINGS),
                  nargs='*')

arguments = pars.parse_args()

d = vars(arguments)
print(d)
