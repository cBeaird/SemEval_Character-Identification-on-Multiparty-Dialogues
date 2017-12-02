#!/usr/bin/env bash
export PYTHONPATH=../../

python ../conll_2_csv.py -m ../friends_word2vec_model -c ../../datasets-None-8c441e63-e82a-48e6-b1a7-07811cc80cd8-friends.train.trial/friends.train.episode_delim.conll -w2v t -p t -o weka.csv
sed -i "s/'//g" weka.csv

java weka.core.converters.CSVLoader weka.csv -B 50000 > weka.arff
java weka.classifiers.trees.J48 -C 0.25 -M 2 -t weka.arff > ../../Results/C45Results.txt
