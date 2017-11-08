import pandas as pd
import tensorflow as tf
import numpy as np

feature_columns = [tf.feature_column.numeric_column("x", shape=[104])]  # List the features
classifier = tf.estimator.DNNClassifier(                                # Load Classifier
    feature_columns=feature_columns,
    hidden_units=[512, 256, 128],
    n_classes=401,
    model_dir='nn_model'
)

test_df = pd.read_csv("testing_vectors.csv")            # Load the test data
testing_labels = test_df[['Entity_ID']]                 # Load the test labels
testing_vectors = test_df.drop(['Entity_ID'], axis=1)   # Load the test vectors
X_test = np.array(testing_vectors.values, np.float)     # Define test vectors as numpy array
y_test = np.array(testing_labels.values, np.int64)      # Define the test labels as numpy array

test_input_fn = tf.estimator.inputs.numpy_input_fn(     # Input function for test data
    x={"x": X_test},
    y=y_test,
    num_epochs=1,
    shuffle=False)

accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]  # Accuracy of DNN Classifier
print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
