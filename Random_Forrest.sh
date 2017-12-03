#!/usr/bin/env bash

echo Random Forrest Classifier
python Classifiers/conll_2_csv.py -m Classifiers/friends_word2vec_model -c datasets-None-8c441e63-e82a-48e6-b1a7-07811cc80cd8-friends.train.trial/friends.train.episode_delim.conll -f -o vectors.csv -w2v
python Classifiers/Random_Forest/Random_Forest.py -tr vectors.csv -o Classifiers/Models/random-forest.pkl
python Classifiers/Random_Forest/eval_rand_forrest.py -te vectors.csv -m Classifiers/Models/random-forest.pkl
rm Classifiers/Models/random-forest.pkl
