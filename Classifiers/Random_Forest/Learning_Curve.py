import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn.learning_curve import learning_curve


def split_labels_and_vectors(csv_path,label_name):
    df = pd.read_csv(csv_path)
    labels = df[label_name].values.tolist()
    vectors = df.drop([label_name], axis=1).values
    return labels,vectors


def plot_curve(X, y, cv):
    lg = RandomForestClassifier(n_jobs=-1, max_features=None, oob_score=True,
                                 n_estimators=63, max_depth=30, min_samples_leaf=1,
                                 )

    lg.fit(X, y)

    train_sizes, train_scores, test_scores = learning_curve(lg, X, y, n_jobs=-1, cv=cv,
                                                            train_sizes=np.linspace(.1, 1.0, 5), verbose=0)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

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

size = 1000
cv = KFold(size, shuffle=True)
labels, vectors = split_labels_and_vectors(csv_path="../vectors.csv", label_name="Entity_ID")
plot_curve(vectors, labels, cv)