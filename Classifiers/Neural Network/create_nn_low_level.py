import tensorflow as tf
import numpy as np
import pandas as pd

n_inputs = 105        # There are 105 features
n_hidden1 = 300       # The first hidden layer will have 300 neurons
n_hidden2 = 100       # The second hidden layer will have 100 neurons
n_outputs = 401       # There are 401 different classes
learning_rate = 0.01  # Learning Rate for the DNN
num_epochs = 1000     # Number of training loops
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")  # Placeholder for the training data
y = tf.placeholder(tf.int64, shape=None, name="y")  # Placeholder for the training labels

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X,n_hidden1,name="hidden1", activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy,name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()   # Initialize all the variables
saver = tf.train.Saver()                   # Saver for the variables

train_df = pd.read_csv("training_vectors.csv")  # Load in the training data
test_df = pd.read_csv("testing_vectors.csv")    # Load in the testing data

training_labels = train_df['Entity_ID']
training_vectors = train_df.drop(["Entity_ID"], axis=1)
testing_labels = test_df['Entity_ID']
testing_vectors = test_df.drop(['Entity_ID'], axis=1)

X_train = np.array(training_vectors.values, np.float)
y_train = np.array(training_labels.values, np.int64)
X_test = np.array(testing_vectors.values, np.float)
y_test = np.array(testing_labels.values, np.int64)

with tf.Session() as sess:
    init.run()
    for epoch in range(num_epochs):
        sess.run(training_op,feed_dict={X: X_train, y: y_train})
        acc_train = accuracy.eval(feed_dict={X: X_train, y: y_train})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
        saver.save(sess, 'nn_model')
