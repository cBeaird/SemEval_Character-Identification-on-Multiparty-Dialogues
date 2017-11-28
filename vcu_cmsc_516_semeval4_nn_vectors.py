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
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.contrib.learn as learn
import numpy as np

__author__ = 'Casey Beaird'
__credits__ = ['Casey Beaird', 'Chase Greco', 'Brandon Watts']
__license__ = 'MIT'
__version__ = '0.1'

PATTERN_FOR_DOC_ID = '(?:\/.*-s)([0-9]*)(?:[a-z])([0-9]*)'
MENTION_FINDER = '(?:^#)(?:.*)([0-9)]$)'

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

# Specifying training is requested (as opposed to testing)
pars.add_argument('-t', '--train',
                  help='Train/extend the model {m} using the {data_file} provided.',
                  action='store_true')

# Specifying training is requested (as opposed to testing)
pars.add_argument('-e', '--evaluate',
                  help='Evaluate the {data_file} using the training model {m} provided.',
                  action='store_true')

# map file this is the entity map file for the named entities
pars.add_argument('-mf', '--mapFile',
                  help='name of entity map the file that contains the entity names and their entity IDs',
                  type=file,
                  dest='map_file')

# training data file
pars.add_argument('-df', '--dataFile',
                  help='Input file for either training or evaluating',
                  type=file,
                  dest='data_file')

# column headers for the conll file if this is not specified then the headings file is assumed to
# be in the format from the SemEval 2018 Task 4 format that is specified in the core model file.
pars.add_argument('-d', '--headers',
                  help='column type list for the order of columns in you training file',
                  dest='columns',
                  choices=list(sEcm.DEFAULT_HEADINGS),
                  default=sEcm.DEFAULT_HEADINGS,
                  nargs='*')

arguments = pars.parse_args()
d = vars(arguments)

# check args for problems
# todo parse the command line arguments this will create the namespace class that gives access to the
# arguments passed in the command line. This will need to be broken out to deal with all the disjoint
# sets of operations we want:
#     1: train and pickle training data
#     2: init the model
#     3: evaluate from the current model
if d['train'] and d['evaluate']:
    print('cannot train  and evaluate at the same time on the same data file!\n')
    exit(0)
# if d['data_file'] is None:
#     print('need a data file to evaluate or train on')
#     # exit(0)

# columns object will always exist because there is a default list of columns so we can set the columns
# the training data uses the Default_headings in the model python file so we dont need to care about dealing
# with in in a specific way
if d['columns'] != sEcm.DEFAULT_HEADINGS:
    sEcm.DEFAULT_HEADINGS = d['columns']

# will need a way to save models
sEcm.nn_model = dict()

# start the NN training
if d['train']:
    if d['map_file'] is not None:
        sEcm.nn_model[sEcm.MODEL_ENTITY_MAP] = sEcf.add_map_file_to_nn_model(d['map_file'])
    if d['data_file'] is not None:
        nn_data = sEcf.train_nn_model(d['data_file'])
        train, evaluate = train_test_split(nn_data[0][1:])
        with open('train_data.csv', 'w') as parsed_data_file:
            # parsed_data_file.write(','.join(str(h) for h in nn_data[0][0]) + '\n')
            for instance in train:
                parsed_data_file.write(','.join(str(c) for c in instance) + '\n')
        with open('evaluate_data.csv', 'w') as parsed_data_file:
            # parsed_data_file.write(','.join(str(h) for h in nn_data[0][0]) + '\n')
            for instance in evaluate:
                parsed_data_file.write(','.join(str(c) for c in instance) + '\n')

    # Tensorflow part
    training_set = learn.datasets.base.load_csv_without_header(filename='test_nn_data_file.csv',
                                                               target_dtype=np.int,
                                                               features_dtype=np.float)

    evaluate_set = learn.datasets.base.load_csv_without_header(filename='train_nn_data_file.csv',
                                                               target_dtype=np.int,
                                                               features_dtype=np.float)

    # Season, Episode, Scene_ID, Speaker, Lemma, POS_Tag,
    # wv_0, wv_1, wv_2, wv_3, wv_4, wv_5, wv_6, wv_7, wv_8, wv_9, wv_10, wv_11, wv_12, wv_13, wv_14, wv_15, wv_16,
    # wv_17, wv_18, wv_19, wv_20, wv_21, wv_22, wv_23, wv_24, wv_25, wv_26, wv_27, wv_28, wv_29, wv_30, wv_31, wv_32,
    # wv_33, wv_34, wv_35, wv_36, wv_37, wv_38, wv_39, wv_40, wv_41, wv_42, wv_43, wv_44, wv_45, wv_46, wv_47, wv_48,
    # wv_49, wv_50, wv_51, wv_52, wv_53, wv_54, wv_55, wv_56, wv_57, wv_58, wv_59, wv_60, wv_61, wv_62, wv_63, wv_64,
    # wv_65, wv_66, wv_67, wv_68, wv_69, wv_70, wv_71, wv_72, wv_73, wv_74, wv_75, wv_76, wv_77, wv_78, wv_79, wv_80,
    # wv_81, wv_82, wv_83, wv_84, wv_85, wv_86, wv_87, wv_88, wv_89, wv_90, wv_91, wv_92, wv_93, wv_94, wv_95, wv_96,
    # wv_97, wv_98, wv_99, Word, Entity_ID

    train_array = np.array(training_set.data)
    season = tf.feature_column.numeric_column('season')
    season_v = train_array[:, 0]
    episode = tf.feature_column.numeric_column('episode')
    episode_v = train_array[:, 1]
    scene = tf.feature_column.numeric_column('scene')
    scene_v = train_array[:, 2]
    speaker = tf.feature_column.numeric_column('speaker')
    speaker_v = train_array[:, 3]
    lemma = tf.feature_column.numeric_column('lemma')
    lemma_v = train_array[:, 4]
    pos = tf.feature_column.numeric_column('pos')
    pos_v = train_array[:, 5]
    # w2v_column = tf.feature_column.categorical_column_with_identity(key='w2v', num_buckets=1000000, default_value=0)
    w2v = tf.feature_column.numeric_column('w2v', 100)  # embedding_column(w2v_column, 100)
    w2v_v = train_array[:, 6:106]
    word = tf.feature_column.numeric_column('word')
    word_v = train_array[:, 106]

    feature_columns = [season, episode, scene, speaker, lemma, pos, w2v, word]
    # feature_columns = [tf.feature_column.numeric_column('instance', [6])]
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[1000, 500, 401],
                                            n_classes=len(sEcm.nn_model[sEcm.MODEL_ENTITY_MAP]),
                                            model_dir='tensorflow_nn_w2v_model')

    Y = np.array(training_set.target)
    train_function = tf.estimator.inputs.numpy_input_fn(x={'season': season_v, 'episode': episode_v, 'scene': scene_v,
                                                           'speaker': speaker_v, 'lemma': lemma_v, 'pos': pos_v,
                                                           'w2v': w2v_v, 'word': word_v},
                                                        # {'instance': np.array(training_set.data)},
                                                        y=Y,
                                                        num_epochs=None,
                                                        shuffle=True)

    classifier.train(input_fn=train_function, steps=20000)

    eval_array = np.array(evaluate_set.data)
    season_ev = eval_array[:, 0]
    episode_ev = eval_array[:, 1]
    scene_ev = eval_array[:, 2]
    speaker_ev = eval_array[:, 3]
    lemma_ev = eval_array[:, 4]
    pos_ev = eval_array[:, 5]
    w2v_ev = eval_array[:, 6:106]
    word_ev = eval_array[:, 106]

    eY = np.array(evaluate_set.target)
    evaluate_function = tf.estimator.inputs.numpy_input_fn(x={'season': season_ev, 'episode': episode_ev,
                                                              'scene': scene_ev, 'speaker': speaker_ev,
                                                              'lemma': lemma_ev, 'pos': pos_ev, 'w2v': w2v_ev,
                                                              'word': word_ev},
                                                           # {'instance': np.array(evaluate_set.data)},
                                                           y=eY,
                                                           num_epochs=1,
                                                           shuffle=False)

    accuracy = classifier.evaluate(input_fn=evaluate_function)['accuracy']

    print(accuracy)
else:
    print('Nothing was asked of me!\nThank you and have a good day!')
exit(0)
