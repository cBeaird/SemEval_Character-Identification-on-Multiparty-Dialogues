import tensorflow as tf
import pandas as pd
import numpy as np

train_df = pd.read_csv("training_vectors.csv")           # Obtain the training data
training_labels = train_df['Entity_ID']                  # Get the labels from the training data
training_vectors = train_df.drop(["Entity_ID"], axis=1)  # Drop the labels from the training data
X_train = np.array(training_vectors.values, np.float)    # Place the feature vectors into a numpy array
y_train = np.array(training_labels.values, np.int64)     # Place the labels into a numpy array

feature_columns = [tf.feature_column.numeric_column("x", shape=[104])]  # All the features are important

classifier = tf.estimator.DNNClassifier(  # Create our DNN Classifier with 3 hidden layers
    feature_columns=feature_columns,
    hidden_units=[512, 256, 128],
    n_classes=401,
    model_dir='nn_model'
)

train_input_fn = tf.estimator.inputs.numpy_input_fn(   # Define the training input for the DNN
    x={"x": X_train},
    y=y_train,
    num_epochs=None,
    shuffle=True
)

classifier.train(input_fn=train_input_fn, steps=20000)  # Train the DNN
