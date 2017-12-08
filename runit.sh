#!/usr/bin/env bash

# AUTHOR: Chase Greco

# Add root project directory to PYTHON path environment variable
export PYTHONPATH=${PYTHONPATH}:.

# Make results directory
mkdir Results

# Build Weka Feature Vectors
echo 'Building Weka Feature Vectors'
python Classifiers/conll_2_csv.py -m Classifiers/friends_word2vec_model -c datasets-None-8c441e63-e82a-48e6-b1a7-07811cc80cd8-friends.train.trial/friends.train.episode_delim.conll -w2v -p -o weka.csv

# Convert csv to arff format
echo 'Converting data to arff format'
sed -i "s/'//g" weka.csv
java weka.core.converters.CSVLoader weka.csv -B 50000 > weka.arff
echo 'Done!'

# Train Models
echo 'Training C45'
java weka.classifiers.trees.J48 -C 0.25 -M 2 -t weka.arff > Results/C45.txt
echo 'Results stored in Results/C45.txt'

# Cleanup
rm ./weka.csv
rm ./weka.arff

# Build NN Feature Vectors
echo 'Building Neural Net Feature Vectors'
python ./Classifiers/conll_2_csv.py -m ./Classifiers//friends_word2vec_model -c ./datasets-None-8c441e63-e82a-48e6-b1a7-07811cc80cd8-friends.train.trial/friends.train.episode_delim.conll -f -w2v -s -o nn_data_file.csv

sed '1d' test_nn_data_file.csv > tpf; mv tpf test_nn_data_file.csv
sed '1d' train_nn_data_file.csv > tpff; mv tpff train_nn_data_file.csv

rm -rf ./tensorflow_nn_w2v_model

mkdir ./tensorflow_nn_w2v_model
echo 'Done!'

# Training Neural Net
echo 'Training Neural Net'
python ./vcu_cmsc_516_semeval4_nn_vectors.py -m friends_nn_ep --train -mf ./datasets-None-8c441e63-e82a-48e6-b1a7-07811cc80cd8-friends.train.trial/friends_entity_map.txt > Results/NeuralNet.txt
echo 'Results stored in Results/NeuralNet.txt'

#Cleanup
rm -rf ./tensorflow_nn_w2v_model
rm test_nn_data_file.csv
rm train_nn_data_file.csv

# Build Random Forest Feature Vectors
echo 'Building Random Forest Feature Vectors'
python Classifiers/conll_2_csv.py -m Classifiers/friends_word2vec_model -c datasets-None-8c441e63-e82a-48e6-b1a7-07811cc80cd8-friends.train.trial/friends.train.episode_delim.conll -f -o vectors.csv -w2v
echo 'Done!'

# Training Random Forest
echo 'Training Random Forest'
python Classifiers/Random_Forest/Random_Forest.py -tr vectors.csv -o Classifiers/Models/random-forest.pkl
python Classifiers/Random_Forest/Random_Forest_Evaluation.py -te vectors.csv -m Classifiers/Models/random-forest.pkl > Results/RandomForest.txt
echo 'Results stored in Results/RandomForest.txt'

#Cleanup
rm Classifiers/Models/random-forest.pkl
rm train_vectors.csv
rm test_vectors.csv
rm vectors.csv