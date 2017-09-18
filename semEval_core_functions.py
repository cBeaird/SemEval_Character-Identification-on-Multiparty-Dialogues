#!/usr/bin/env python
"""
Core functions for conference resolution for SemEval 2018 task 4

This file houses the core functionality required by the application. This file holds all
the defined functions and classes the application needs to train and evaluate the input
data to perform the entity identification task.

"""
import os
import pickle
import re
import semEval_core_model as sEcm
from conllu.parser import parse

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
    :rtype: dict
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
    # type: (probability_matrix, speakers, words) -> tuple
    """
    Build basic probability matrix with the speaker the word and the entity with the associated counts for
    who they are referring to. This is not a very pythonic way to build this dict of dicts but it is easy to
    follow.

    example: {'Monica_Geller' {'he' {'count' = 10, 222 = 5, 111 = 5}}}

    :param data_file: training data file
    :rtype: tuple
    :return: tuple with speakers, words and dictionary of dictionaries with the counts for referred entities
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

                    probability_matrix[speaker][word]['count'] = probability_matrix[speaker][word].get('count', 0) + 1
                    probability_matrix[speaker][word][eid] = probability_matrix[speaker][word].get(eid, 0) + 1

            except KeyError:
                continue
    return probability_matrix, words, speakers


def update_model(model_item_name, model_object):
    """
    sets/updates the model object to the corresponding entry in the model dictionary
    Note!: any item in the model will need to be extended in the model file and a corresponding entry
    needs to be build.
    :param model_item_name: model dict key
    :param model_object: object to be set in model
    """
    if sEcm.model is None:
        sEcm.model = dict()

    sEcm.updater_functions[model_item_name](model_object)


def load_model(model_name):
    # type: (bool) -> bool
    """
    load the pickled model from the model directory models are keyed by name
    :param model_name: name of model to load
    :rtype: bool
    :return: model was loaded
    """
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
    """
    pickle and write the model to the model directory
    :param model_name:
    """
    with open('./model/{}'.format(model_name), 'wb') as mf:
        pickle.dump(sEcm.model, mf)


def update_entities(entities):
    """
    update the entities if the entity map grows beyond the original set
    :param entities: entities dict(id, name)
    """
    if sEcm.MODEL_ENTITY_MAP in sEcm.model:
        for key in entities.keys():
            sEcm.model[sEcm.MODEL_ENTITY_MAP][key] = entities[key]
    else:
        sEcm.model[sEcm.MODEL_ENTITY_MAP] = entities


def update_speakers(speakers):
    """
    update the speakers set
    :param speakers: set of speakers
    """
    if sEcm.MODEL_SPEAKERS in sEcm.model:
        for speaker in speakers:
            sEcm.model[sEcm.MODEL_SPEAKERS].add(speaker)
    else:
        sEcm.model[sEcm.MODEL_SPEAKERS] = speakers


def update_words(words):
    """
    update the words set
    :param words: set of words
    """
    if sEcm.MODEL_WORDS in sEcm.model:
        for word in words:
            sEcm.model[sEcm.MODEL_WORDS].add(word)
    else:
        sEcm.model[sEcm.MODEL_WORDS] = words


def update_dist_counts(counts):
    """
    update the distribution counts for all references
    :param counts: dict of dicts
    """
    if sEcm.MODEL_DISTRIBUTIONS in sEcm.model:
        # todo this ones is slightly more complicated but need to add new words entities and add those counts
        sEcm.model[sEcm.MODEL_DISTRIBUTIONS] = counts
    else:
        sEcm.model[sEcm.MODEL_DISTRIBUTIONS] = counts


class ConllWord:
    pattern_for_document_id = None

    def __init__(self, **kwargs):
        self.doc_id = None
        self.scene_id = None
        self.token_id = None
        self.word = None
        self.pos = None
        self.constituency = None
        self.lemma = None
        self.frame_id = None
        self.ws = None
        self.speaker = None
        self.ne = None
        self.e_id = None
        self.extra_items = dict()

        for k, v in kwargs.iteritems():
            if hasattr(self, k):
                vars(self)[k] = v
            else:
                self.extra_items[k] = v

    def __str__(self):
        return ("Document ID: {doc_id}\nScene ID: {scene_id}\nToken ID: {token_id}\n"
                "Word: {word}\n""POS: {pos}\nConstituency Tag: {constituency}\nLemma: {lemma}\n"
                "Frameset ID: {frame_id}\n""Word Sense: {ws}\nSpeaker: {speaker}\nNamed Entity: {ne}\n"
                "Entity ID: {e_id}\n".format(doc_id=self.doc_id, scene_id=self.scene_id, token_id=self.token_id,
                                             word=self.word, pos=self.pos,
                                             constituency=self.constituency, lemma=self.lemma, frame_id=self.frame_id,
                                             ws=self.ws, speaker=self.speaker,
                                             ne=self.ne, e_id=self.e_id))

    @staticmethod
    def define_doc_contents(pattern):
        ConllWord.pattern_for_document_id = re.compile(pattern)
