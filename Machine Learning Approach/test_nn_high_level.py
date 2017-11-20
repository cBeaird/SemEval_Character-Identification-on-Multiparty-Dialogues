import pandas as pd
import tensorflow as tf
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import seaborn as sn

pars = argparse.ArgumentParser(usage='Evaluates DNN',
                               formatter_class=argparse.RawTextHelpFormatter,
                               description='''Evaluates DNN for Semeval''',
                               version='0.1')

pars.add_argument('-m', '--model',
                  help='Directory for the DNN Model')
pars.add_argument('-te', '--testing',
                  help='Filepath to the testing CSV')


def load_classifier(model_directory):
    feature_columns = [tf.feature_column.numeric_column("x", shape=[105])]
    return tf.estimator.DNNClassifier(
           feature_columns=feature_columns,
           hidden_units=[512, 256, 128],
           n_classes=401,
           model_dir=model_directory)


def split_labels_and_vectors(csv_path,label_name):
    df = pd.read_csv(csv_path)
    labels = df[label_name].values.tolist()
    vectors = df.drop([label_name], axis=1).values
    return labels,vectors


def make_predictions(classifier, test_data):
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(  # Define the input function
        x={"x": test_data},
        num_epochs=1,
        shuffle=False)
    predictions = list(classifier.predict(input_fn=predict_input_fn))  # Get the predictions
    predicted_classes = [p["classes"] for p in predictions]  # Extract the predicted classes from the predictions
    predicted_classes = np.array(predicted_classes, np.int32)  # Place the predicted classes in a numpy array
    return predicted_classes.flatten().tolist()  # Convert to List


def main():
    arguments = pars.parse_args()
    args = vars(arguments)
    clf = load_classifier(args['model'])
    labels, vectors = split_labels_and_vectors(csv_path=args["testing"], label_name="Entity_ID")
    preds = make_predictions(clf, vectors)
    print('\nMetrics\n')
    print(classification_report(labels, preds))
    print('\nConfusion Matrix\n')
    df_confusion = confusion_matrix(labels, preds, labels=range(400))
    df_cm = pd.DataFrame(df_confusion)
    print df_cm
    mask = df_cm < 1  # Showing all values greater than 1
    sn.heatmap(df_cm, mask=mask)
    plt.show()


if __name__ == "__main__":
    main()

