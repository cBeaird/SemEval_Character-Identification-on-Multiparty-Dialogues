from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.externals import joblib


def split_labels_and_vectors(csv_path,label_name):
    df = pd.read_csv(csv_path)
    labels = df[label_name].values.tolist()
    vectors = df.drop([label_name], axis=1).values
    return labels,vectors


def main():
    labels, vectors = split_labels_and_vectors(csv_path="../train_vectors.csv", label_name="Entity_ID")
    clf = RandomForestClassifier(n_jobs=-1, max_features='sqrt', oob_score=True,
                                 n_estimators=63, max_depth=30, min_samples_leaf=1,
                                 )
    clf.fit(vectors, labels)
    joblib.dump(clf, '../Models/random-forrest.pkl')


if __name__ == "__main__":
    main()