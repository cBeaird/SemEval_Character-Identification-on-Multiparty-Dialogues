#!/usr/bin/env python
"""
Core functions for conference resolution for SemEval 2018 task 4

This file houses the core functionality required by the application. This file holds all
the defined functions and classes the application needs to train and evaluate the input
data to perform the entity identification task.

"""
# todo need to come up with a way to define the structure of the dict that holds the 'probability matrix'
import os
import pickle
import re
import operator
import semEval_core_model as sEcm
from conllu.parser import parse
import pandas as pd
from random import shuffle

__author__ = 'Casey Beaird'
__credits__ = ['Casey Beaird', 'Chase Greco', 'Brandon Watts']
__license__ = 'MIT'
__version__ = '0.1'

MENTION_FINDER = '(?:^[^#])(?:.*)([0-9)]$)'


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


def translate_file_to_object_list(data_file):
    # type: (object_list) -> list
    """
    reads in the conll data file and returns a list of the objects in file order where
    each object is a word in the file.
    :param data_file: conll file
    :return: list of ConllWord
    :rtype: list
    """
    if not isinstance(data_file, file):
        raise TypeError
    data_file.seek(0)

    object_list = list()
    for line in data_file:
        parse_line = parse(line, sEcm.DEFAULT_HEADINGS)
        if parse_line[0]:
            try:
                parsed_word = parse_line[0][0]
                object_list.append(ConllWord(**parsed_word))
            except (KeyError, ImportError):
                continue
    return object_list


def get_probability_matrix(object_list):
    # type: (speakers, reference_words, decomposition_dict) -> tuple
    """
    better way to build the "probability matrix". we first parse the file into conll objects then
    from those object return a dictionary of the references.
    :param object_list: list of conll objects
    :return: tuple of speakers, words, and the dict with decomposition objects.
    """
    # todo finish building this out
    speakers = set()
    reference_words = set()
    decomposition_dict = dict()
    for word in object_list:
        if word.e_id is not sEcm.EMPTY:
            speakers.add(word.speaker)
            reference_words.add(word.word)
            decomposition_dict[word.speaker] = decomposition_dict.get(word.speaker, CorpusDistributions(word.speaker))
            decomposition_dict[word.speaker].add_word(word.word, word.e_id)

    return speakers, reference_words, decomposition_dict


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
    data_file.seek(0)

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


def conll_2_dataframe(conll_text):
    '''
    BRANDON WATTS
    Method to turn a conll file to a pandas dataframe
    :param conll_text: the conll text
    :return: the connl data in the form of a list of connlWords
    '''
    parsed_text = parse(conll_text, sEcm.DEFAULT_HEADINGS)
    doc_ids = []
    scene_ids = []
    token_ids = []
    words = []
    posTags = []
    posTags_expanded = []
    lemmatizedWords = []
    frameSetIDs = []
    wordSenses = []
    speakers = []
    namedEntities = []
    entityIDs = []

    for sentence in parsed_text:
        for word in sentence:
            doc_ids.append(word.get(sEcm.DOCUMENT_ID, None))
            scene_ids.append(word.get(sEcm.SCENE_ID, None))
            token_ids.append(word.get(sEcm.TOKEN_ID, None))
            words.append(word.get(sEcm.WORD, None).lower())
            posTags.append(word.get(sEcm.POS, None))
            posTags_expanded.append(word.get(sEcm.CONSTITUENCY, None))
            lemmatizedWords.append(word.get(sEcm.LEMMA, None))
            frameSetIDs.append(word.get(sEcm.FRAMESET_ID, None))
            wordSenses.append(word.get(sEcm.WORD_SENSE, None))
            speakers.append(word.get(sEcm.SPEAKER, None))
            namedEntities.append(word.get(sEcm.NE, None))
            entityIDs.append(word.get(sEcm.ENTITY_ID, None))

    df = pd.DataFrame.from_dict({'Document_ID': doc_ids, 'Scene_ID': scene_ids, "Token_ID": token_ids,
                                 "Word": words, "POS_Tag": posTags, "POS_Tag_Expanded": posTags_expanded,
                                 "Lemma": lemmatizedWords, "Frameset_ID": frameSetIDs, "Word_Sense": wordSenses,
                                 "Speaker": speakers, "Named_Entity": namedEntities, "Entity_ID": entityIDs})
    # Extract the Season and episode data from "Document ID" and place it in format: season,episode
    df["Document_ID"] = df["Document_ID"].str.replace(r'\/friends-s(\d+)e(\d+)',
                                                      r'\1,\2')
    extracted_doc_data = df["Document_ID"].apply(lambda x: x.split(','))  # Split the previous format by comma
    df['Season'] = extracted_doc_data.apply(lambda x: x[0])  # Place the Season into its own column
    df['Episode'] = extracted_doc_data.apply(
        lambda x: x[1])  # Place the Episode into its own columnToken ID", "Word Sense", "Lemma",

    return df


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
        # todo this ones is slightly more complicated but need to add new words entities
        # todo and add those counts. probably need a way to define the structure of the dict
        sEcm.model[sEcm.MODEL_DISTRIBUTIONS] = counts
    else:
        sEcm.model[sEcm.MODEL_DISTRIBUTIONS] = counts


def evaluate(data_file):
    # type: (answers) -> list
    """
    evaluate the file based on the model and return a list of answers
    :param data_file: eval file
    :return: list of answers
    """
    if not isinstance(data_file, file):
        raise TypeError
    mention_matcher = re.compile(MENTION_FINDER)
    data_file.seek(0)
    number = number_correct = 0.0
    answers = list()

    for line in data_file:
        clean_line = line.rstrip()
        if len(clean_line) == 0:
            continue
        # if clean_line[-1] is not sEcm.EMPTY:
        if mention_matcher.match(clean_line) is not None:
            parsed_line = parse(line, sEcm.DEFAULT_HEADINGS)
            if parsed_line[0]:
                try:
                    number += 1
                    parsed_word = parsed_line[0][0]
                    c_word = ConllWord(**parsed_word)
                    this_speaker = sEcm.model[sEcm.MODEL_DISTRIBUTIONS].get(c_word.speaker)
                    if this_speaker is not None:
                        if c_word.word in this_speaker:
                            t = this_speaker[c_word.word]
                            t['count'] = 0
                            answer = max(t.iteritems(), key=operator.itemgetter(1))[0]
                            answers.append(answer)
                            if answer == c_word.e_id:
                                number_correct += 1
                            print('{},{}'.format(c_word.e_id, answer))
                        else:
                            # we guess 248 for all the answers we have no idea about
                            print('{},{}'.format(c_word.e_id, '(248)'))
                    else:
                        # we guess 248 for all the answers we have no idea about
                        print('{},{}'.format(c_word.e_id, '(248)'))
                except (KeyError, ImportError):
                    print('KeyError')
                    continue
    print('total: {}, total correct: {}, accuracy: {:2f}'.format(number, number_correct, number_correct / number))
    return answers


def split_raw_conll_file(d_file):
    """
    shuffle and return the raw conll data as a list used for k-fold validation
    :param file: file path where conll file is
    :return: list of conll data one line per list item
    """
    data = list()
    with open(d_file, 'r') as conll_file:
        data = conll_file.readlines()
    shuffle(data)
    return data


''' Neural Network model functions beginning 
############################################
'''


def train_nn_model(data_file):
    """
    non-word to vector csv generation for training the neural network bags each instance
    :param data_file: conll data file
    :return: tuple of instances the word/speaker/pos tag sets.
    """
    feature_header = ['season', 'episode', 'word', 'pos', 'lemma', 'speaker', 'class']
    word_dict = dict()
    speaker_dict = dict()
    pos_tag_dict = dict()
    instance_list = list()
    regex = re.compile(r'[\t ]+')
    ConllWord.define_doc_contents('(?:\/.*-s)([0-9]*)(?:[a-z])([0-9]*)')
    instance_list.append(feature_header)

    for line in data_file:
        new_line = regex.sub('\t', line.lower())
        content = parse(new_line, sEcm.DEFAULT_HEADINGS)

        if content[0]:
            try:
                word = ConllWord(**content[0][0])  # create word object
                word_dict[word.word] = word_dict.get(word.word, len(word_dict) + 1)
                word_dict[word.lemma] = word_dict.get(word.lemma, len(word_dict) + 1)
                speaker_dict[word.speaker] = speaker_dict.get(word.speaker, len(speaker_dict) + 1)
                pos_tag_dict[word.pos] = pos_tag_dict.get(word.pos, len(pos_tag_dict) + 1)

                if word.is_reference():
                    instance_list.append([word.get_document_id_item(1), word.get_document_id_item(2),
                                          word_dict[word.word],
                                          pos_tag_dict[word.pos],
                                          word_dict[word.lemma],
                                          speaker_dict[word.speaker],
                                          word.get_entity_class()])
            except (KeyError, ImportError):
                print('error: {}'.format(new_line))
                continue

    return instance_list, word_dict, speaker_dict, pos_tag_dict


def add_map_file_to_nn_model(map_file):
    """
    using the same model structure as the naive implementation add the new map file to the model (not really needed
    as it turns out tensorflow stores it's own model).
    :param map_file: map file.
    :return:  list of entities in the map file
    """
    entities = dict()
    for line in map_file:
        components = line.rstrip().split()
        if components:
            entities[components[0]] = components[1]
    return entities


''' Neural Network model functions beginning 
############################################
'''


class ConllWord:
    pattern_for_document_id = None
    pattern_for_entity_ref = re.compile(r'(?:\(?)([0-9]{1,3})(?:\)?)')

    def __init__(self, **kwargs):
        """
        constructor build initial state all fields in the core model are in the class additional
        fields can be added and will be added to the extra items dict by name.
        :param kwargs:
        """
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

                # # find and fix the parser errors and fix those we can there are two common classes of these errors
                # if self.e_id is None:
                #     parser_error = self.speaker.split(' ')
                #     if len(parser_error) == 2:
                #         self.speaker = parser_error[0]
                #         self.e_id = parser_error[1]
                #     elif len(parser_error) == 1 and parser_error[0] == '*':
                #         self.e_id = self.ne
                #         self.ne = self.speaker
                #         self.speaker = self.ws[1:]
                #         self.ws = sEcm.EMPTY
                #     else:
                #         # there are ~50 really jacked up entries so if we find them just ignore those lines
                #         raise ImportError

    def __str__(self):
        """
        return string representation of object.
        :return: string of a conll word
        """
        return ("Document ID: {doc_id}\nScene ID: {scene_id}\nToken ID: {token_id}\n"
                "Word: {word}\n""POS: {pos}\nConstituency Tag: {constituency}\nLemma: {lemma}\n"
                "Frameset ID: {frame_id}\n""Word Sense: {ws}\nSpeaker: {speaker}\nNamed Entity: {ne}\n"
                "Entity ID: {e_id}\n".format(doc_id=self.doc_id, scene_id=self.scene_id, token_id=self.token_id,
                                             word=self.word, pos=self.pos,
                                             constituency=self.constituency, lemma=self.lemma, frame_id=self.frame_id,
                                             ws=self.ws, speaker=self.speaker,
                                             ne=self.ne, e_id=self.e_id))

    def __repr__(self):
        """
        return string representation of object
        :return: string of conll word
        """
        return str(self)

    def is_reference(self):
        if ConllWord.pattern_for_entity_ref.match(self.e_id) is None:
            return False
        else:
            return True

    def get_entity_class(self):
        m = ConllWord.pattern_for_entity_ref.match(self.e_id)
        if m is not None:
            return m.group(1)
        else:
            return -1

    def get_document_id_item(self, item):
        # type: (object) -> str
        """
        get information from the document id attribute requires the pattern_for_document_id to be set
        with a regular expression that parses the desired content from the doc_id
        :param item: group requested from the matcher
        :return: contents from the group
        """
        results = ConllWord.pattern_for_document_id.match(self.doc_id)
        return results.group(item)

    def get_feature_representation(self, feature_name):
        """
        returns an attributes hash value this is a naive way of encoding the features
        :param feature_name: feature name
        :return: hash representation
        """
        if not hasattr(self, feature_name):
            return 0
        if feature_name == sEcm.DOCUMENT_ID:
            return int(self.get_document_id_item(1) + self.get_document_id_item(2))

        return hash(vars(self)[feature_name])

    @staticmethod
    def define_doc_contents(pattern):
        """
        sets the class definition of the doc id pattern parser
        :param pattern: regular expression pattern
        """
        ConllWord.pattern_for_document_id = re.compile(pattern)


class CorpusDistributions:
    def __init__(self, speaker, main_character=False):
        self.speaker = speaker
        self.words = dict()
        self.is_main = main_character

    def add_word(self, word, entity):
        self.words[word] = self.words.get(word, dict())
        self.words[word][entity] = self.words[word].get(entity, 0) + 1
