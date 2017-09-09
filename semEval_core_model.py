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
__author__ = 'Casey Beaird'
__credits__ = ['Casey Beaird', 'Chase Greco', 'Brandon Watts']
__license__ = 'MIT'
__version__ = '0.1'

DOCUMENT_ID = 'doc_id'
SCENE_ID = 'scene_id'
TOKEN_ID = 'token_id'
WORD = 'word'
POS = 'pos'
CONSTITUENCY = 'constituency'
LEMMA = 'lemma'
FRAMESET_ID = 'frame_id'
WORD_SENSE = 'ws'
SPEAKER = 'speaker'
NE = 'ne'
ENTITY_ID = 'e_id'

DEFAULT_HEADINGS = (DOCUMENT_ID, SCENE_ID, TOKEN_ID, WORD, POS, CONSTITUENCY,
                    LEMMA, FRAMESET_ID, WORD_SENSE, SPEAKER, NE, ENTITY_ID)

# name key'd dictionary for entity ID's and string names
entity_map = None
