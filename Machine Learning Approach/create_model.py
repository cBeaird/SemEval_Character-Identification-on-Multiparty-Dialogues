import semEval_core_model as sEcm
import sys
import semEval_core_functions as sEcf
import gensim
import numpy as np

__credits__ = ['Casey Beaird', 'Chase Greco', 'Brandon Watts']
__license__ = 'MIT'
__version__ = '0.1'

def openFile(filepath):
    with open(filepath, 'r') as conll_file:
        conll_file_text = conll_file.read()
    return conll_file_text

path_to_data = sys.argv[1]                                            # Location of the Conll File
conll_file = openFile(path_to_data)
df = sEcf.conll_2_dataframe(conll_file)                               # Put Conll File into a pandas dataframe
df = df[['Word']]                                                     # Delete every column besides the words
df['Word'] = df['Word'].str.replace(r'^[\.\?]|!{1,2}|\?!$', '<END>')  # Replace the end of sentences with the <END> tag
df = df.replace(r'^[,\'-\*]$|^\.+$', np.nan, regex=True)                  # Replace all the junk with NaN
df = df.dropna()                                                      # Drop all of the junk

text = []                           # List that will hold the format gensim needs
sentence_text = []                  # Temporary list to hold words in a sentence
for word in df['Word']:             # Loop through all the words
    if word == "<END>":             # If it is an <END> Tag add the sentence to the text and start another sentence
        text.append(sentence_text)
        sentence_text = []
    else:                           # If it is not the end of the sentence keep adding words to sentence text
        sentence_text.append(word)

model = gensim.models.Word2Vec(text, min_count=1)
model.save('friends_word2vec_model')
