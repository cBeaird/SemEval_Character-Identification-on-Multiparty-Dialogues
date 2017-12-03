#!/usr/bin/env python
"""
Script used to create a Random Forrest
"""

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.externals import joblib
import argparse

__author__ = 'Brandon Watts'
__credits__ = ['Casey Beaird', 'Chase Greco']
__license__ = 'MIT'
__version__ = '0.1'


pars = argparse.ArgumentParser(usage='Creates a Random Forest Classifier',
                               formatter_class=argparse.RawTextHelpFormatter,
                               description='''Creates a Random Forest Classifier for Semeval''',
                               version='0.1')

pars.add_argument('-tr', '--train',
                  help='Path to the training CSV file')

pars.add_argument('-o', '--output',
                  help='Path to the Model output')


def split_labels_and_vectors(csv_path, label_name):
    """
    Method used to split a csv into two dataframes: labels and vectors
    :param csv_path: Path to the CSV file
    :param label_name: Name of the label column
    :return: label and vector dataframes
    """
    df = pd.read_csv(csv_path)
    labels = df[label_name].values.tolist()
    vectors = df.drop([label_name], axis=1).values
    return labels,vectors


def main():
    arguments = pars.parse_args()
    args = vars(arguments)
    labels, vectors = split_labels_and_vectors(csv_path=args["train"], label_name="Entity_ID")
    clf = RandomForestClassifier(n_jobs=-1, max_features=None, oob_score=True, n_estimators=63,
                                 max_depth=30, min_samples_leaf=1)
    clf.fit(vectors, labels)
    joblib.dump(clf, args["output"])


if __name__ == "__main__":
    main()
