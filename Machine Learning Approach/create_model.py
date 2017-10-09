import semEval_core_model as sEcm
from conllu.parser import parse
import re
import sys
import semEval_core_functions as sEcf
import gensim

'''
This class is used to create the word2vec model called "friends_word2vec_model".
'''
__credits__ = ['Casey Beaird', 'Chase Greco', 'Brandon Watts']
__license__ = 'MIT'
__version__ = '0.1'

def main():
    '''
    Simple script to create a word2vec model from the supplied input as a conll file
    '''

    text = [] # Holds the text in the format that gensim wants
    path_to_data = sys.argv[1] # location of the conll file

    with open(path_to_data, 'r') as conllFile:
        conll_text = conllFile.read()
    parsed_conll = parseConll(conll_text)

    # Loop over the conll file and place in in the following format:
    # [[sentence1,sentence1,...],[sentence2,sentence2,...],..]
    for sentence in parsed_conll:
        sentence_text = []
        for word in sentence:
            if re.match('\'?\w+', word.word):    # If the item we are looking at is a word
                sentence_text.append(word.word)  # Put it in a sentence
        text.append(sentence_text)               # Add that sentence to our text list

    # Create the word2vec model and save it for later use
    model = gensim.models.Word2Vec(text, min_count=1)
    model.save('friends_word2vec_model')

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
            except ImportError:
                continue
        connlWords.append(connlSentence)
    return connlWords

if __name__ == '__main__':
    main()
