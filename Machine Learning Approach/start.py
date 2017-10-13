import semEval_core_model as sEcm
import sys
import semEval_core_functions as sEcf
import gensim
import pandas as pd
import numpy as np

__credits__ = ['Casey Beaird', 'Chase Greco', 'Brandon Watts']
__license__ = 'MIT'
__version__ = '0.1'

word2vec_model = gensim.models.Word2Vec.load('friends_word2vec_model')

def openFile(filepath):
    with open(filepath, 'r') as conll_file:
        conll_file_text = conll_file.read()
    return conll_file_text

conll_file = openFile(sys.argv[1])                          # Open Conll File
df = sEcf.conll_2_dataframe(conll_file)                     # Obtain data frame
df = df.replace(r'[-]', np.nan, regex=True)                 # Replace all the entity ids that are dashes wil NaN
df['Entity ID'] = df['Entity ID'].str.replace(r'\D+', ' ')  # Delete all of the parentheis around the numbers
df['Entity ID'] = df['Entity ID'].str.strip()               # Strip extra spaces
df = df.drop(["POS Tag", "POS Tag Expanded", "Named Entity", "Frameset ID",
                  "Token ID", "Word Sense", "Lemma", "Document ID"], axis=1)  # Drop unused columns
df = df.dropna(subset=['Entity ID'])                                          # Drop all the Entity ID's = NaN
df["Speaker"] = pd.factorize(df['Speaker'])[0]                                # Place Speakers into numerical form
dataframe_filter = df['Word'].str.contains(r'\w+')
df = df[dataframe_filter]                                                 # Remove all the mentions that are not words
df["Word"] = df["Word"].apply(lambda x: word2vec_model[x].tolist())       # Change word to Word2Vec Representation
word_vectors = df['Word'].apply(pd.Series)                                # Place word2vec values in their own columns
word_vectors = word_vectors.rename(columns=lambda x: 'wv_' + str(x))      # Rename the columns from wv_0..wv_n
appended_data = pd.concat([df[["Season", "Episode",                       # Add the word vectors to our dataframe
                               "Scene ID", "Speaker"]], word_vectors[:]], axis=1)
df = pd.concat([appended_data[:], df["Entity ID"]], axis=1)               # Add the labels back on
df.to_csv("weka.csv", index=False)                                        # Place dataframe in a csv file