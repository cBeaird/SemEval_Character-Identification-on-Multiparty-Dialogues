import argparse
import gensim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
sys.path.append("./")
import semEval_core_model as sEcm
import semEval_core_functions as sEcf

__credits__ = ['Casey Beaird', 'Chase Greco', 'Brandon Watts']
__license__ = 'MIT'
__version__ = '0.1'

pars = argparse.ArgumentParser(usage='Turn a Conll File in to a CSV',
                               formatter_class=argparse.RawTextHelpFormatter,
                               description='''Any Conll File to CSV''',
                               version='0.1')

pars.add_argument('-m', '--model',
                  help='Directory for the word2vec model')
pars.add_argument('-c', '--conll',
                  help='Path of the Connll File')
pars.add_argument('-f', '--factorize',
                  help='Boolean argument on rather to turn strings to numerical form',
                  action='store_true')
pars.add_argument('-w2v', '--word2vec',
                  help='Boolean argument on rather to use Word2Vec',
                  action='store_true')
pars.add_argument("-s", '--split',
                  help='Boolean argument on rather to split into training & testing CSVs',
                  action='store_true')
pars.add_argument("-o", '--output',
                  help='Output file name')
pars.add_argument("-p", '--paren',
                  help='Boolean argument on whether to include parentheses around class',
                  action='store_true')


def open_file(filepath):
    with open(filepath, 'r') as conll_file:
        conll_file_text = conll_file.read()
    return conll_file_text


def cleanDF(df):
    df = df.replace(r'[-\*]', np.nan, regex=True)  # Replace all the entity ids that are dashes wil NaN
    df['Entity_ID'] = df['Entity_ID'].str.replace(r'\D+', ' ')  # Delete all of the parentheis around the numbers
    df['Entity_ID'] = df['Entity_ID'].str.strip()  # Strip extra spaces
    df = df.drop(["Document_ID", "POS_Tag_Expanded", "Named_Entity", "Word_Sense", "Frameset_ID"], axis=1)
    df = df.dropna(subset=['Entity_ID'])  # Drop all the Entity ID's = NaN
    dataframe_filter = df['Word'].str.contains(r'\w+')
    df = df[dataframe_filter]  # Remove all the mentions that are not words
    return df


def factorize(df):
    df["Speaker"] = pd.factorize(df['Speaker'])[0]  # Place Speakers into numerical form
    df["Lemma"] = pd.factorize(df['Lemma'])[0]  # Place Speakers into numerical form
    df["POS_Tag"] = pd.factorize(df['POS_Tag'])[0]  # Place Speakers into numerical form
    df["Word"] = pd.factorize(df['Word'])[0]  # Place Speakers into numerical form
    return df


def word2vec(df, model):
    word2vec_model = gensim.models.Word2Vec.load(model)
    temp_word_column = df[["Word"]]
    df["Word"] = df["Word"].apply(lambda x: word2vec_model[x].tolist())  # Change word to Word2Vec Representation
    word_vectors = df['Word'].apply(pd.Series)  # Place word2vec values in their own columns
    word_vectors = word_vectors.rename(columns=lambda x: 'wv_' + str(x))  # Rename the columns from wv_0..wv_n
    appended_data = pd.concat([df[["Season", "Episode", "Scene_ID", "Speaker", "Lemma", "POS_Tag"]], word_vectors[:]],
                              axis=1)
    appended_data = pd.concat([appended_data, temp_word_column], axis=1)
    df = pd.concat([appended_data[:], df["Entity_ID"]], axis=1)  # Add the labels back on
    return df


def paren(df):
    df["Entity_ID"] = df["Entity_ID"].apply(lambda c: '(' + c + ')')
    return df


def main():
    arguments = pars.parse_args()
    args = vars(arguments)
    word2vec_model_loc = args["model"]
    conll_file = open_file(args["conll"])
    df = sEcf.conll_2_dataframe(conll_file)
    df = cleanDF(df)
    if args["word2vec"]:
        df = word2vec(df, word2vec_model_loc)
    if args["paren"]:
        df = paren(df)
    if args["factorize"]:
        df = factorize(df)
    if args["split"]:
        train_set, test_set = train_test_split(df, test_size=.2, random_state=42)
        train_set.to_csv("train_" + args["output"], index=False)
        test_set.to_csv("test_" + args["output"], index=False)
    else:
        df.to_csv(args["output"], index=False)


if __name__ == "__main__":
    main()
