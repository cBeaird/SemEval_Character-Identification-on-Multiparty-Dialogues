from unittest import TestCase
import semEval_core_model
import semEval_core_functions
from semEval_core_functions import ConllWord


class TestConllWord(TestCase):
    conll_word = ConllWord(doc_id='/friends-s01e8876', scene_id='0',
                           token_id='0', word='There',
                           pos='EX', constituency='(TOP(S(NP*)',
                           lemma='there', frame_id='-',
                           ws='-', speaker='Monica_Geller',
                           ne='*', e_id='-')

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
