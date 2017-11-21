import pandas as pd
from sklearn.externals import joblib
from sklearn import metrics


def split_labels_and_vectors(csv_path,label_name):
    df = pd.read_csv(csv_path)
    labels = df[label_name].values.tolist()
    vectors = df.drop([label_name], axis=1).values
    return labels,vectors

def main():

    labelName = "Entity_ID"
    labels, vectors = split_labels_and_vectors(csv_path="test_vectors.csv", label_name="Entity_ID")

    clf = load_classifier()
    preds = clf.predict(vectors)
    targs = labels

    print_metrics(targs, preds)


def load_classifier():
    return joblib.load('Models/random-forrest.pkl')


def print_metrics(targs, preds):
    print('\nMetrics\n')
    print("accuracy: ", metrics.accuracy_score(targs, preds ))
    print("precision: ", metrics.precision_score(targs, preds, average="weighted"))
    print("recall: ", metrics.recall_score(targs, preds, average="weighted"))
    print("f1: ", metrics.f1_score(targs, preds, average="weighted"))
    print("Geometric Mean", metrics.fowlkes_mallows_score(targs, preds))


if __name__ == "__main__":
    main()
