#!/usr/bin/env python
"""
File to plot the Learning Curve of a Random Forrest
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib as mpl
from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve
mpl.use('TkAgg')
import matplotlib.pyplot as plt

__author__ = 'Brandon Watts'
__credits__ = ['Casey Beaird', 'Chase Greco']
__license__ = 'MIT'
__version__ = '0.1'


def split_labels_and_vectors(csv_path,label_name):
    """
    Method used to split a csv into two dataframes: labels and vectors
    :param csv_path: Path to the CSV file
    :param label_name: Name of the label column
    :return: label and vector dataframes
    """
    df = pd.read_csv(csv_path)
    df_labels = df[label_name].values.tolist()
    df_vectors = df.drop([label_name], axis=1).values
    return df_labels, df_vectors


def plot_curve(x, y, folds):
    """
    Method used to plot the Learning Curve
    :param x: vectors
    :param y: labels
    :param folds: how many folds for cross-validation
    """

    # Create and Train a classifier
    classifier = RandomForestClassifier(n_jobs=-1, max_features=None, oob_score=True,
                                        n_estimators=63, max_depth=30, min_samples_leaf=1)
    classifier.fit(x, y)

    # Create the Learning Curve for the Classifier
    train_sizes, train_scores, test_scores = learning_curve(classifier, x, y, n_jobs=-1, cv=folds,
                                                            train_sizes=np.linspace(.1, 1.0, 5), verbose=0)

    # Extract all the stats for the plot
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Create the plot
    plt.figure()
    plt.title("RandomForestClassifier")
    plt.legend(loc="best")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.gca().invert_yaxis()
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.ylim(-.1, 1.1)
    plt.show()


cv = KFold(10, shuffle=True)
labels, vectors = split_labels_and_vectors(csv_path="../../vectors.csv", label_name="Entity_ID")
plot_curve(vectors, labels, cv)
