import semEval_core_model as sEcm
import sys
import semEval_core_functions as sEcf
import gensim
import pandas as pd
import numpy as np

__credits__ = ['Casey Beaird', 'Chase Greco', 'Brandon Watts']
__license__ = 'MIT'
__version__ = '0.1'


def openFile(filepath):
    with open(filepath, 'r') as conll_file:
        conll_file_text = conll_file.read()
    return conll_file_text


word2vec_model = gensim.models.Word2Vec.load('friends_word2vec_model')
path_to_data = sys.argv[1]                                                # Location of the Conll File
conll_file = openFile(path_to_data)
df = sEcf.conll_2_dataframe(conll_file)
df = df.replace(r'[-]', np.nan, regex=True)                               # Replace all the entity ids that are dashes wil NaN
df['Entity_ID'] = df['Entity_ID'].str.replace(r'\D+', ' ')                # Delete all of the parentheis around the numbers
df['Entity_ID'] = df['Entity_ID'].str.strip()                             # Strip extra spaces
df = df.drop(["POS_Tag", "POS_Tag_Expanded", "Named_Entity",
              "Frameset_ID","Token_ID", "Word_Sense",
              "Lemma", "Document_ID"], axis=1)  # Drop unused columns
df = df.dropna(subset=['Entity_ID'])                                      # Drop all the Entity ID's = NaN
df["Speaker"] = pd.factorize(df['Speaker'])[0]                            # Place Speakers into numerical form
dataframe_filter = df['Word'].str.contains(r'\w+')
df = df[dataframe_filter]                                                 # Remove all the mentions that are not words
df["Word"] = df["Word"].apply(lambda x: word2vec_model[x].tolist())       # Change word to Word2Vec Representation
word_vectors = df['Word'].apply(pd.Series)                                # Place word2vec values in their own columns
word_vectors = word_vectors.rename(columns=lambda x: 'wv_' + str(x))      # Rename the columns from wv_0..wv_n
appended_data = pd.concat([df[["Season", "Episode",                       # Add the word vectors to our dataframe
                               "Scene_ID", "Speaker"]],
                              word_vectors[:]], axis=1)
df = pd.concat([appended_data[:], df["Entity_ID"]], axis=1)               # Add the labels back on

df.to_csv("weka.csv", index=False)                                        # Place dataframe in a csv file
