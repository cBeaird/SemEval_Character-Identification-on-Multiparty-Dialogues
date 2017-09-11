#!/usr/bin/env python
"""
Core functions for conference resolution for SemEval 2018 task 4

This file houses the core functionality required by the application. This file holds all
the defined functions and classes the application needs to train and evaluate the input
data to perform the entity identification task.

"""
from conllu.parser import parse
import semEval_core_model as sEcm

__author__ = 'Casey Beaird'
__credits__ = ['Casey Beaird', 'Chase Greco', 'Brandon Watts']
__license__ = 'MIT'
__version__ = '0.1'


def build_entity_dict(map_file):
    # type: (entity_map) -> dict
    """
    Entity dictionary builder, build the entity dictionary form the command line file or any other file
    that contains a tab delimited (entity_ID, entity_name) list of entities in the corpora.

    :param map_file: entity map file
    :return: dictionary of entity id's and string names
    """
    if not isinstance(map_file, file):
        raise TypeError

    entity_map = dict()
    for line in map_file:
        line_items = line.rstrip().split('\t')
        entity_map[line_items[0]] = line_items[1]

    return entity_map


def build_basic_probability_matrix(data_file):
    # type: (probability_matrix) -> dict
    """
    Build basic probability matrix with the speaker the word and the entity with the associated counts for
    who they are referring to. This is not a very pythonic way to build this dict of dicts but it is easy to
    follow.

    example: {'Monica_Geller' {'he' {'count' = 10, 222 = 5, 111 = 5}}}

    :param data_file: training data file
    :return: dictionary of dictionaries with the counts for referred entities
    """
    if not isinstance(data_file, file):
        raise TypeError

    # this is Janky need to figure out key error and gracefully continue
    # also clearly not finished
    probability_matrix = dict()
    for line in data_file:
        p = parse(line, sEcm.DEFAULT_HEADINGS)
        if p[0]:
            try:
                eid = p[0][0][sEcm.ENTITY_ID]
                speaker = p[0][0][sEcm.SPEAKER]
                word = p[0][0][sEcm.WORD]

                if eid != sEcm.EMPTY:
                    if speaker not in probability_matrix:
                        probability_matrix[speaker] = dict()
                        probability_matrix[speaker][word] = dict()
                        probability_matrix[speaker][word]['count'] = 0
                    if word not in probability_matrix[speaker]:
                        probability_matrix[speaker][word] = dict()
                        probability_matrix[speaker][word]['count'] = 0
                    if eid not in probability_matrix[speaker][word]:
                        probability_matrix[speaker][word][eid] = 0
                    probability_matrix[speaker][word]['count'] += 1
                    probability_matrix[speaker][word][eid] += 1

            except KeyError:
                break
    return probability_matrix
