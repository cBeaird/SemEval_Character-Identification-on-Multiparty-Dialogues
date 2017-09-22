import semEval_core_model as sEcm
from conllu.parser import parse
import csv
import re
import sys
import semEval_core_model as sEcm
import semEval_core_functions as sEcf

__author__ = 'Brandon Watts'
__credits__ = ['Casey Beaird', 'Chase Greco', 'Brandon Watts']
__license__ = 'MIT'
__version__ = '0.1'

EntityMap = {}
ENTITY_COUNT = 0

def main():

    path_to_data = sys.argv[1]
    with open(path_to_data, 'r') as myfile:
        conll_text = myfile.read()
    trainingData = parseConll(conll_text)
    featureVectors = createFeatureVectors(trainingData)
    writeCSV(featureVectors)

def parseConll(conll_text):
    '''
    Stolen from semEval_core_fuctions. Will use that method in near future.
    '''
    parsed_text = parse(conll_text, sEcm.DEFAULT_HEADINGS)
    connlWords = []
    for sentence in parsed_text:
        connlSentence = []
        for word in sentence:
            sEcf.ConllWord.define_doc_contents('(?:\/.*-s)([0-9]*)(?:[a-z])([0-9]*)')
            cwd = sEcf.ConllWord(doc_id=word.get(sEcm.DOCUMENT_ID, None), scene_id=word.get(sEcm.SCENE_ID, None),
                            token_id=word.get(sEcm.TOKEN_ID, None), word=word.get(sEcm.WORD, None),
                            pos=word.get(sEcm.POS, None), constituency=word.get(sEcm.CONSTITUENCY, None),
                            lemma=word.get(sEcm.LEMMA, None), frame_id=word.get(sEcm.FRAMESET_ID, None),
                            ws=word.get(sEcm.WORD_SENSE, None), speaker=word.get(sEcm.SPEAKER, None),
                            ne=word.get(sEcm.NE, None), e_id=word.get(sEcm.ENTITY_ID, None))
            connlSentence.append(cwd)
        connlWords.append(connlSentence)

    return connlWords

def createFeatureVectors(trainingData):
    '''
    Method to create feature vectors from the parsed connl text.
    Feature vectors are in the following format: fv_i = [Season Number, Episode Number, Speaker, The given word's id, Enitity it is reffering to]
    where featureVectors = [[fv_1],[fv_2],...,[fv_n]]
    :param trainingData: The parsed connl text
    :return: An array of feature vectors
    '''
    SEASON = 1
    EPISODE = 2
    featureVectors = []
    for sentence in trainingData:
        for word in sentence:
            if containsReference(word):
                addEntity(word)
                featureVectors.append([word.get_document_id_item(SEASON),word.get_document_id_item(EPISODE), word.speaker, EntityMap[word.word], re.match(r'\((\d+)\)', word.e_id).group(1)])
    return featureVectors

def containsReference(word):
    '''
    Helper method to check if a word contains a reference
    :param word: connlWord
    :return: Boolean value rather it contains a reference or not
    '''
    if word.e_id == None: # Will figure out why it is coming back as 'None' but for now this works
        return False
    else:
        return re.match(r'\(\d+\)', word.e_id)

def addEntity(entity):
    '''
    Method to create our entity map (Not sure if we are keeping this)
    :param entity: A given entity such as "He" or "Mother"
    '''
    if not EntityMap.has_key(entity):
        global ENTITY_COUNT
        ENTITY_COUNT += 1
        EntityMap[entity.word] = ENTITY_COUNT

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
