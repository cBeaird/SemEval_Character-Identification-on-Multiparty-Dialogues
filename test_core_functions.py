from unittest import TestCase
import os
import semEval_core_model as sEcm
import semEval_core_functions as sEcf
from semEval_core_functions import ConllWord


class TestConllWord(TestCase):
    conll_word = None
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

    @classmethod
    def setUpClass(cls):
        print('Setting testing objects\n')
        TestConllWord.conll_word = ConllWord(doc_id='/friends-s01e8876', scene_id='0',
                                             token_id='0', word='There',
                                             pos='EX', constituency='(TOP(S(NP*)',
                                             lemma='there', frame_id='-',
                                             ws='-', speaker='Monica_Geller',
                                             ne='*', e_id='-')
        with open('./testTemp', 'wb') as tf:
            tf.write(TestConllWord.conll_text)

    @classmethod
    def tearDownClass(cls):
        print('Testing completed\n')
        TestConllWord.conll_word = None
        os.remove('./testTemp')

    def test_get_document_id_item(self):
        # /friends-s01e8876|0|0|There|EX|(TOP(S(NP*)|there|-|-|Monica_Geller|*|-
        ConllWord.define_doc_contents('(?:\/.*-s)([0-9]*)(?:[a-z])([0-9]*)')

        self.assertEquals(TestConllWord.conll_word.get_document_id_item(1), '01')
        self.assertEquals(TestConllWord.conll_word.get_document_id_item(2), '8876')

    def test_conll_word_print(self):
        self.assertEquals(str(TestConllWord.conll_word), 'Document ID: /friends-s01e8876\n'
                                                         'Scene ID: 0\n'
                                                         'Token ID: 0\n'
                                                         'Word: There\n'
                                                         'POS: EX\n'
                                                         'Constituency Tag: (TOP(S(NP*)\n'
                                                         'Lemma: there\n'
                                                         'Frameset ID: -\n'
                                                         'Word Sense: -\n'
                                                         'Speaker: Monica_Geller\n'
                                                         'Named Entity: *\n'
                                                         'Entity ID: -\n')

    def test_get_feature_rep(self):
        # doc id is treated special it's the int of the episode and scene concatenated
        self.assertEquals(TestConllWord.conll_word.get_feature_representation(sEcm.DOCUMENT_ID), int('018876'))

        # everything is just the hash of the object
        self.assertEquals(hash('There'), TestConllWord.conll_word.get_feature_representation(sEcm.WORD))
        self.assertEquals(hash('Monica_Geller'), TestConllWord.conll_word.get_feature_representation(sEcm.SPEAKER))

    def test_word_to_object(self):
        obj_list = None
        with open('./testTemp', 'r') as tf:
            obj_list = sEcf.translate_file_to_object_list(tf)

        self.assertIsNotNone(obj_list)

    def test_object_decomp(self):
        obj_list = None
        with open('./testTemp', 'r') as tf:
            obj_list = sEcf.translate_file_to_object_list(tf)

        speaker_set, decom = sEcf.get_probability_matrix(obj_list)
        self.assertEqual(len(speaker_set), 1)
        self.assertIsNotNone(decom.get('Monica_Geller'))
