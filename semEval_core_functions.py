#!/usr/bin/env python
"""
Core functions for conference resolution for SemEval 2018 task 4

This file houses the core functionality required by the application. This file holds all
the defined functions and classes the application needs to train and evaluate the input
data to perform the entity identification task.

"""
import os
import pickle
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
    # type: (probability_matrix) -> Tuple[dict, set, set]
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

    # build initial data objects for the speakers and words list and the dict of use
    speakers = set()
    words = set()
    probability_matrix = dict()

    for line in data_file:
        p = parse(line, sEcm.DEFAULT_HEADINGS)
        if p[0]:
            try:
                eid = p[0][0][sEcm.ENTITY_ID]
                speaker = p[0][0][sEcm.SPEAKER]
                word = p[0][0][sEcm.WORD]
                if eid != sEcm.EMPTY:
                    words.add(word)
                    speakers.add(speaker)
                    if speaker not in probability_matrix:
                        probability_matrix[speaker] = dict()
                        probability_matrix[speaker][word] = dict()
                    if word not in probability_matrix[speaker]:
                        probability_matrix[speaker][word] = dict()

                    probability_matrix[speaker][word]['count'] = probability_matrix[speaker][word].get('count', 0)+1
                    probability_matrix[speaker][word][eid] = probability_matrix[speaker][word].get(eid, 0)+1

            except KeyError:
                continue
    return probability_matrix, words, speakers


def update_model(model_item_name, model_object):
    if sEcm.model is None:
        sEcm.model = dict()

    sEcm.model[model_item_name] = model_object


def load_model(model_name):
    if not os.path.isdir('./model'):
        os.mkdir('./model')
    if not os.path.isfile('./model/{}'.format(model_name)):
        open('./model/{}'.format(model_name), 'w').close()
        sEcm.model_path = './model/{}'.format(model_name)
        return False

    sEcm.model_path = './model/{}'.format(model_name)
    with open(sEcm.model_path, 'rb') as mf:
        sEcm.model = pickle.load(mf)
    return True


def save_model(model_name):
    with open('./model/{}'.format(model_name), 'wb') as mf:
        pickle.dump(sEcm.model, mf)
    return True
