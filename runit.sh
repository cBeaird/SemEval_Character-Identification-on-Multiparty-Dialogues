#!/usr/bin/env bash

# AUTHOR: Chase Greco

# Add root project directory to PYTHON path environment variable
export PYTHONPATH=${PYTHONPATH}:.

# Make results directory
mkdir Results

# Build probabilistic model
echo 'Training Simple Probabilistic Model'
python vcu_cmsc516_semeval4.py -m friendsEp --train -mf ./datasets-None-8c441e63-e82a-48e6-b1a7-07811cc80cd8-friends.train.trial/friends_entity_map.txt -df ./datasets-None-8c441e63-e82a-48e6-b1a7-07811cc80cd8-friends.train.trial/friends.train.episode_delim.conll
echo 'Evaluating Simple Probabilistic Model'
python vcu_cmsc516_semeval4.py -m friendsEp --evaluate -df ./datasets-None-8c441e63-e82a-48e6-b1a7-07811cc80cd8-friends.train.trial/friends.train.episode_delim.conll > ./Results/ProbabilisticResults.txt
echo 'Results stored in Results/ProbabilisticResults.txt'

# Build Word2Vec Model
echo 'Training Word2Vec Model'
cd Machine\ Learning\ Approach/
python create_model.py ../datasets-None-8c441e63-e82a-48e6-b1a7-07811cc80cd8-friends.train.trial/friends.train.episode_delim.conll

# Build Feature vectors
echo 'Building Feature Vectors'
python start.py ../datasets-None-8c441e63-e82a-48e6-b1a7-07811cc80cd8-friends.train.trial/friends.train.episode_delim.conll ../datasets-None-8c441e63-e82a-48e6-b1a7-07811cc80cd8-friends.train.trial/friends_entity_map.txt

# Convert csv to arff format
echo 'Converting data to arff format'
java weka.core.converters.CSVLoader weka.csv -B 50000 > weka.arff

# Train Models
echo 'Training Naive Bayes'
java weka.classifiers.bayes.NaiveBayes -t weka.arff > ../Results/NaiveBayesResults.txt
echo 'Results stored in Results/NaiveBayesResults.txt'

echo 'Training C45'
java weka.classifiers.trees.J48 -C 0.25 -M 2 -t weka.arff > ../Results/C45Results.txt
echo 'Results stored in Results/C45Results.txt'

echo 'Training SVM'
java -Xmx6g weka.classifiers.functions.SMO -C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K "weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007" -calibrator "weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4" -t weka.arff > ../Results/SVMResults.txt
echo 'Results stored in Results/SVMResults.txt'

echo 'Calculating the Mutual Information (Gain Ratio) for each attribute'
java weka.attributeSelection.GainRatioAttributeEval -i weka.arff > ../Results/MutualInfoResults.txt
echo 'Results stored in Results/MutualInfoResults.txt'