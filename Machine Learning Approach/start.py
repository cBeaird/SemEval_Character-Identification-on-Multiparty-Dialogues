import tensorflow as tf
import numpy as np
import sys
import semEval_core_model as sEcm
from conllu.parser import parse
from semEval_core_functions import ConllWord
import gensim
from nltk.corpus import brown

# Simple text for testing consists of top two lines of the data
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

WINDOW_SIZE = 5
sentences = brown.sents()
model = gensim.models.Word2Vec(sentences, min_count=1)
model.save("brown_model")

def main():
    trainingData = test_conll_parse(conll_text)
    createFeatureVectors(trainingData)

def getWordVector(word):
    return model[word]


def test_conll_parse(conll_text):
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
    for sentence in trainingData:
        if containsReference(sentence):
            print (getWordVector(sentence[0].word))

def containsReference(sentence):
    containsReference = False
    for word in sentence:
        if word.e_id != "-":
            containsReference = True
    return containsReference

if __name__ == '__main__':
    main()
