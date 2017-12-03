#!/usr/bin/env bash

echo Random Forrest Classifier
python Classifiers/conll_2_csv.py -m Classifiers/friends_word2vec_model -c datasets-None-8c441e63-e82a-48e6-b1a7-07811cc80cd8-friends.train.trial/friends.train.episode_delim.conll -f -s -o vectors.csv -w2v
python Classifiers/Random_Forest/random_forrest_classifier.py -tr train_vectors.csv -o Classifiers/Models/random-forest.pkl
python Classifiers/Random_Forest/eval_rand_forrest.py -te test_vectors.csv -m Classifiers/Models/random-forest.pkl
rm Classifiers/Models/random-forest.pkl
rm train_vectors.csv
rm test_vectors.csv
