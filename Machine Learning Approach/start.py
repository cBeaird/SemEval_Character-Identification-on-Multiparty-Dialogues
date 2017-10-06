import semEval_core_model as sEcm
from conllu.parser import parse
import csv
import re
import sys
import semEval_core_functions as sEcf
import gensim
from feature_vector import feature_vector

__author__ = 'Brandon Watts'
__credits__ = ['Casey Beaird', 'Chase Greco', 'Brandon Watts']
__license__ = 'MIT'
__version__ = '0.1'

speakers = {}  # Dictionary of speakers

def main():

    # Open the conll file and parse data from it
    path_to_data = sys.argv[1]
    with open(path_to_data, 'r') as myfile:
        conll_text = myfile.read()
    data = parseConll(conll_text)

    word2vec_model = gensim.models.Word2Vec.load('friends_word2vec_model') # Load the word2vec model
    featureVectors = createFeatureVectors(data,word2vec_model) # Create our feature vectors
    writeCSV(featureVectors, 100) # Write feature vectors to csv file TODO: Don't hardcode the model length

def parseConll(conll_text):
    '''
    BRANDON WATTS
    Method to parse the conll data from a conll file
    :param conll_text: the conll text
    :return: the connl data in the form of a list of connlWords
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
            except ImportError:
                continue
        connlWords.append(connlSentence)

    return connlWords

def createFeatureVectors(trainingData,word2vec_model):
    '''
    BRANDON WATTS
    Method to create feature vectors from the parsed connl text.
    where featureVectors = [[fv_1],[fv_2],...,[fv_n]]
    :param trainingData: The parsed connl text
    :param word2vec_model: The genism word2vec model
    :return: An array of feature vectors
    '''
    SEASON = 1
    EPISODE = 2
    feature_vectors = []
    for sentence in trainingData:
        for word in sentence:
            if containsReference(word) and re.match(r"\w+",word.word):
                fv = feature_vector(season_id =  '(' + word.get_document_id_item(SEASON) + ')',
                                    episode_id = '(' + word.get_document_id_item(EPISODE) + ')',
                                    speaker_id = '(' + repr(getSpeakerNumber(word.speaker)) + ')',
                                    word_vector = word2vec_model[word.word],
                                    e_id = word.e_id
                                    )
                feature_vectors.append(fv.get_vector_representation())
    return feature_vectors

def getSpeakerNumber(entity):
    '''
    BRANDON WATTS
    Method to get the unique id of a speaker
    :param entity: The name of the entity
    :return: Numerical ID of the entity
    '''
    global speakers
    if entity not in speakers:
        speakers[entity] = len(speakers) + 1
    return speakers[entity]

def containsReference(word):
    '''
    BRANDON WATTS
    Helper method to check if a word contains a reference
    :param word: connlWord
    :return: Boolean value rather it contains a reference or not
    '''
    return re.match(r'\(\d+\)', word.e_id)

def writeCSV(featureVectors, modelLength):
    '''
    BRANDON WATTS
    Method to create our csv file so that chase can perform WEKA magic
    :param featureVectors: Array of feature vectors
    '''
    csvfile = "weka.csv"
    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')

        # Build header for top of csv
        header = ["season_id", "episode_id", "speaker_id"]
        for x in range(modelLength):
                header.append("a" + repr(x))
        header.append("class")

        writer.writerow(header)

        writer.writerows(featureVectors)

if __name__ == '__main__':
    main()
