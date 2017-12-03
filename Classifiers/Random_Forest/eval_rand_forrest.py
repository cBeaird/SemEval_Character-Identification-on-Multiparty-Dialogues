import pandas as pd
from sklearn.externals import joblib
from sklearn import metrics
import warnings
import argparse

pars = argparse.ArgumentParser(usage='Creates a Random Forest Classifier',
                               formatter_class=argparse.RawTextHelpFormatter,
                               description='''Creates a Random Forest Classifier for Semeval''',
                               version='0.1')

pars.add_argument('-te', '--test',
                  help='Path to the testing CSV file')

pars.add_argument('-m', '--model',
                  help='Path to the Model')


def split_labels_and_vectors(csv_path,label_name):
    df = pd.read_csv(csv_path)
    labels = df[label_name].values.tolist()
    vectors = df.drop([label_name], axis=1).values
    return labels,vectors


def main():
    warnings.warn = warn
    arguments = pars.parse_args()
    args = vars(arguments)
    labelName = "Entity_ID"
    labels, vectors = split_labels_and_vectors(csv_path=args['test'], label_name="Entity_ID")

    clf = joblib.load(args['model'])
    preds = clf.predict(vectors)
    targs = labels

    print_metrics(targs, preds)


def print_metrics(targs, preds):
    print('\nMetrics\n')
    print("accuracy: ", metrics.accuracy_score(targs, preds ))
    print("precision: ", metrics.precision_score(targs, preds, average="weighted"))
    print("recall: ", metrics.recall_score(targs, preds, average="weighted"))
    print("f1: ", metrics.f1_score(targs, preds, average="weighted"))
    print("Geometric Mean", metrics.fowlkes_mallows_score(targs, preds))
    print("Kappa", metrics.cohen_kappa_score(targs, preds))


def warn(*args, **kwargs):
    pass

if __name__ == "__main__":
    main()
