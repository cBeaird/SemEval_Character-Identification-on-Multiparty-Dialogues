#!/usr/bin/env python
"""
Core functions for conference resolution for SemEval 2018 task 4

This file houses the core functionality required by the application. This file holds all
the defined functions and classes the application needs to train and evaluate the input
data to perform the entity identification task.

"""
import semEval_core_model as sEcm

__author__ = 'Casey Beaird'
__credits__ = ['Casey Beaird', 'Chase Greco', 'Brandon Watts']
__license__ = 'MIT'
__version__ = '0.1'


def build_entity_dict(mapFile):
    if not isinstance(mapFile, file):
        raise TypeError

    entity_map = dict()
    for line in mapFile:
        line_items = line.rstrip().split('\t')
        entity_map[line_items[1]] = line_items[0]

    return entity_map
