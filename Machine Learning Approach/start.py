import semEval_core_model as sEcm
from conllu.parser import parse
from semEval_core_functions import ConllWord
import csv
import re

EntityMap = {}
ENTITY_COUNT = 0

# Simple text for testing consists of top two lines of the data (will be removed)
conll_text = """#begin document (/friends-s01e01)
/friends-s01e01   0   0           There    EX               (TOP(S(NP*)           there     -     -   Monica_Geller          *          -
/friends-s01e01   0   1              's   VBZ                      (VP*              be     -     -   Monica_Geller          *          -
/friends-s01e01   0   2         nothing    NN                      (NP*         nothing     -     -   Monica_Geller          *          -
/friends-s01e01   0   3              to    TO                    (S(VP*              to     -     -   Monica_Geller          *          -
/friends-s01e01   0   4            tell    VB                 (VP*)))))            tell     -     -   Monica_Geller          *          -
/friends-s01e01   0   5               !     .                       *))               !     -     -   Monica_Geller          *          -

/friends-s01e01   0   0              He   PRP               (TOP(S(NP*)              he     -     -   Monica_Geller          *      (284)
/friends-s01e01   0   1              's   VBZ                      (VP*              be     -     -   Monica_Geller          *          -
/friends-s01e01   0   2            just    RB                   (ADVP*)            just     -     -   Monica_Geller          *          -
/friends-s01e01   0   3            some    DT                   (NP(NP*            some     -     -   Monica_Geller          *          -
/friends-s01e01   0   4             guy    NN                        *)             guy     -     -   Monica_Geller          *      (284)
/friends-s01e01   0   5               I   PRP              (SBAR(S(NP*)               I     -     -   Monica_Geller          *      (248)
/friends-s01e01   0   6            work   VBP                      (VP*            work     -     -   Monica_Geller          *          -
/friends-s01e01   0   7            with    IN                (PP*))))))            with     -     -   Monica_Geller          *          -
/friends-s01e01   0   8               !     .                       *))               !     -     -   Monica_Geller          *          -"""

def main():
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
            ConllWord.define_doc_contents('(?:\/.*-s)([0-9]*)(?:[a-z])([0-9]*)')
            cwd = ConllWord(doc_id=word.get(sEcm.DOCUMENT_ID, None), scene_id=word.get(sEcm.SCENE_ID, None),
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
    return word.e_id != "-"

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
