import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def create_feature_colums(df):
    headers = list(df)
    feature_columns = []
    for header in headers:
        feature_columns.append(tf.feature_column.numeric_column(header))
    return feature_columns


df = pd.read_csv("weka.csv")  # Place contents of CSV file into a dataframe
train_set, test_set = train_test_split(df, test_size=.2, random_state=42)  # Split the data into test and training data

########## Training the NN ##########

training_set = train_set.copy()                        # Make a copy of the training data
feature_columns = create_feature_colums(training_set)  # Create feature columns

dnn_clf = tf.estimator.DNNClassifier(                  # Deep NN Classifier
    feature_columns=feature_columns,
    hidden_units=[300, 100],
    n_classes=401)

labels = training_set['Entity_ID']  # Classes

train_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=training_set,
    y=labels,
    batch_size=100,
    num_epochs=100,
    shuffle=True)

dnn_clf.train(input_fn=train_input_fn, steps=2000)

########## Testing ##########

testing_set = test_set.copy()
labels = testing_set['Entity_ID']

test_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=testing_set,
    y=labels,
    num_epochs=1,
    shuffle=True
)

accuracy_score = dnn_clf.evaluate(input_fn=test_input_fn)["accuracy"]

print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
