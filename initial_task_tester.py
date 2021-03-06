import os
import tensorflow
import semEval_core_model as sEcm
import semEval_core_functions as sEcf
from conllu.parser import parse
from nltk.corpus.reader import conll
from semEval_core_functions import ConllWord

# """ Author Casey Beaird set up for parsing provided data and ensuring that
# all the needed packages are installed and ready for use."""
__author__ = "Casey Beaird"

# Document ID: /<name of the show>-<season ID><episode ID> (e.g., /friends-s01e01).
# Scene ID: the ID of the scene within the episode.
# Token ID: the ID of the token within the sentence.
# Word form: the tokenized word.
# Part-of-speech tag: the part-of-speech tag of the word (auto generated).
# Constituency tag: the Penn Treebank style constituency tag (auto generated).
# Lemma: the lemma of the word (auto generated).
# Frameset ID: not provided (always "_").
# Word sense: not provided (always "_").
# Speaker: the speaker of this sentence.
# Named entity tag: the named entity tag of the word (auto generated).
# Entity ID: the entity ID of the mention, that is consistent across all documents.
SEMEVAL_FILE_COLUMNS = ('doc_id', 'scene_id', 'token_id', 'word', 'pos', 'con_tag', 'lemma',
                        'frameset_id', 'ws', 'speaker', 'named_entity',
                        'entity_id')

# Columns provided in the data by Conll type best i can tell
column_data = [conll.ConllCorpusReader.CHUNK, conll.ConllCorpusReader.CHUNK,
               conll.ConllCorpusReader.WORDS, conll.ConllCorpusReader.POS,
               conll.ConllCorpusReader.CHUNK, conll.ConllCorpusReader.CHUNK,
               conll.ConllCorpusReader.IGNORE, conll.ConllCorpusReader.IGNORE,
               conll.ConllCorpusReader.NE, conll.ConllCorpusReader.NE,
               conll.ConllCorpusReader.NE]

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


# ID             part  word#         word  POS                  parse bit          lemma  frame sense

# build simple tensor and verify that Tensorflow has been installed
def test_tensor():
    hello = tensorflow.constant('hello, TF')
    sess = tensorflow.Session()
    print(sess.run(hello))

# build conll parser from Conll package verify that conll is installed
# will clean up later - Brandon
def test_conll_parse():
    parsed_text = parse(conll_text, sEcm.DEFAULT_HEADINGS)
    connlWords = []
    for sentence in parsed_text:
        for word in sentence:
            ConllWord.define_doc_contents('(?:\/.*-s)([0-9]*)(?:[a-z])([0-9]*)')

            cwd = ConllWord(doc_id=word.get(sEcm.DOCUMENT_ID, None), scene_id=word.get(sEcm.SCENE_ID, None),
                            token_id=word.get(sEcm.TOKEN_ID, None), word=word.get(sEcm.WORD, None),
                            pos=word.get(sEcm.POS, None), constituency=word.get(sEcm.CONSTITUENCY, None),
                            lemma=word.get(sEcm.LEMMA, None), frame_id=word.get(sEcm.FRAMESET_ID, None),
                            ws=word.get(sEcm.WORD_SENSE, None), speaker=word.get(sEcm.SPEAKER, None),
                            ne=word.get(sEcm.NE, None), e_id=word.get(sEcm.ENTITY_ID, None))
            print(cwd)


# build nltk conll reader to verify that nltk is installed
def test_nltk_parse():
    path_to_data = os.path.dirname(os.path.abspath('__file__')) + \
                   '/datasets-None-8c441e63-e82a-48e6-b1a7-07811cc80cd8-friends.train.trial/' + \
                   'friends.train.scene_delim.conll'
    reader = conll.ConllCorpusReader(path_to_data, [path_to_data], column_data)
    reader.ensure_loaded()
    print(reader)

# perform package installation test
test_tensor()
test_conll_parse()
test_nltk_parse()

print('All imports are correct')
exit(0)
