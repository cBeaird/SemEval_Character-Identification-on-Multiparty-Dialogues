import tensorflow as tf
import pandas as pd
import argparse
import numpy as np

pars = argparse.ArgumentParser(usage='Creates a DNN Classifier',
                               formatter_class=argparse.RawTextHelpFormatter,
                               description='''Creates a DNN Classifier for Semeval''',
                               version='0.1')

pars.add_argument('-tr', '--train',
                  help='Directory for the testing CSV Data')
pars.add_argument('-l', '--labels',
                  help='Name of the labels')


def main():
    arguments = pars.parse_args()
    args = vars(arguments)
    train_df = pd.read_csv(args["train"])                       # Obtain the training data
    training_labels = train_df[args["labels"]]                  # Get the labels from the training data
    training_vectors = train_df.drop([args["labels"]], axis=1)  # Drop the labels from the training data
    X_train = np.array(training_vectors.values, np.float)       # Place the feature vectors into a numpy array
    y_train = np.array(training_labels.values, np.int64)        # Place the labels into a numpy array
    number_of_features = X_train.shape[1]

    feature_columns = [tf.feature_column.numeric_column("x", shape=[number_of_features])]

    classifier = tf.estimator.DNNClassifier(  # Create our DNN Classifier with 3 hidden layers
        feature_columns=feature_columns,
        hidden_units=[512, 256, 128],
        n_classes=401,
        model_dir='nn_model'
    )

    train_input_fn = tf.estimator.inputs.numpy_input_fn(  # Define the training input for the DNN
        x={"x": X_train},
        y=y_train,
        num_epochs=None,
        shuffle=True
    )

    classifier.train(input_fn=train_input_fn, steps=20000)  # Train the DNN


if __name__ == "__main__":
    main()
