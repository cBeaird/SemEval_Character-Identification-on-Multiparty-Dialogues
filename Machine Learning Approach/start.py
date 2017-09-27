import semEval_core_model as sEcm
from conllu.parser import parse
import csv
import re
import sys
import semEval_core_functions as sEcf
import gensim
import numpy

__author__ = 'Brandon Watts'
__credits__ = ['Casey Beaird', 'Chase Greco', 'Brandon Watts']
__license__ = 'MIT'
__version__ = '0.1'

def main():

    path_to_data = sys.argv[1] # File path to conll data
    with open(path_to_data, 'r') as myfile:
        conll_text = myfile.read()
    data = parseConll(conll_text)

    word2vec_model = gensim.models.Word2Vec.load('friends_word2vec_model') # Load the word2vec model
    featureVectors = createFeatureVectors(data,word2vec_model) # Create our feature vectors
    writeCSV(featureVectors) # Write feature vectors to csv file

def parseConll(conll_text):
    '''
    Simple method to parse conll file
    '''
    parsed_text = parse(conll_text, sEcm.DEFAULT_HEADINGS)
    connlWords = []
    for sentence in parsed_text:
        connlSentence = []
        for word in sentence:
            try:
                sEcf.ConllWord.define_doc_contents('(?:\/.*-s)([0-9]*)(?:[a-z])([0-9]*)')
                cwd = sEcf.ConllWord(doc_id=word.get(sEcm.DOCUMENT_ID, None), scene_id=word.get(sEcm.SCENE_ID, None),
                            token_id=word.get(sEcm.TOKEN_ID, None), word=word.get(sEcm.WORD, None).lower(),
                            pos=word.get(sEcm.POS, None), constituency=word.get(sEcm.CONSTITUENCY, None),
                            lemma=word.get(sEcm.LEMMA, None), frame_id=word.get(sEcm.FRAMESET_ID, None),
                            ws=word.get(sEcm.WORD_SENSE, None), speaker=word.get(sEcm.SPEAKER, None),
                            ne=word.get(sEcm.NE, None), e_id=word.get(sEcm.ENTITY_ID, None))
                connlSentence.append(cwd)
            except:
                continue
        connlWords.append(connlSentence)

    return connlWords

def createFeatureVectors(trainingData,word2vec_model):
    '''
    Method to create feature vectors from the parsed connl text.
    Feature vectors are in the following format: fv_i = [Season Number, Episode Number, Speaker, The given word's id, Enitity it is reffering to]
    where featureVectors = [[fv_1],[fv_2],...,[fv_n]]
    :param trainingData: The parsed connl text
    :return: An array of feature vectors
    '''
    SEASON = 1
    EPISODE = 2
    feature_vectors = []
    for sentence in trainingData:
        for word in sentence:
            if containsReference(word):
                feature_vectors.append([word.get_document_id_item(SEASON),word.get_document_id_item(EPISODE), word.speaker, word2vec_model[word.word],word.e_id])
    return feature_vectors

def containsReference(word):
    '''
    Helper method to check if a word contains a reference
    :param word: connlWord
    :return: Boolean value rather it contains a reference or not
    '''
    return re.match(r'\(\d+\)', word.e_id)

def writeCSV(featureVectors):
    '''
    Method to create our csv file so that chase can perform WEKA magic
    :param featureVectors: Array of feature vectors
    '''
    csvfile = "weka.csv"
    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(featureVectors)

if __name__ == '__main__':
    main()
