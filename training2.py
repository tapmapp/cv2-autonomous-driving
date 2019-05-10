import tensorflow as tf
from tensorflow import keras
import numpy as np

EPOCHS = 10
BATCH_SIZE = 16



train_data = (
    np.array([
        [0.1, 0.1],
        [0.2, 0.1],
        [0.3, 0.1],
        [0.4, 0.1],
        [0.1, 0.1],
        [0.2, 0.1],
        [0.3, 0.1],
        [0.4, 0.1],
        [0.1, 0.1],
        [0.2, 0.1],
        [0.3, 0.1],
        [0.4, 0.1],
        [0.1, 0.1],
        [0.2, 0.1],
        [0.3, 0.1],
        [0.4, 0.1],
        [0.5, 0.1]
    ]), 
    np.array([
        [0.1],
        [0.2],
        [0.3],
        [0.4],
        [0.1],
        [0.2],
        [0.3],
        [0.4],
        [0.1],
        [0.2],
        [0.3],
        [0.4],
        [0.1],
        [0.2],
        [0.3],
        [0.4],
        [0.5]
    ]))

test_data = (
    np.array([
        [0.1, 0.1],
        [0.2, 0.1]
    ]), 
    np.array([
        [0.1],
        [0.2]
    ]))


# using two numpy arrays
features, labels = (train_data[0], train_data[1])
dataset = tf.data.Dataset.from_tensor_slices((features,labels)).repeat().batch(BATCH_SIZE)
iter = dataset.make_one_shot_iterator()
x, y = iter.get_next()

# make a simple model
net = tf.layers.dense(x, 8, activation=tf.tanh) 

# pass the first value from iter.get_next() as input
net = tf.layers.dense(net, 8, activation=tf.tanh)
prediction = tf.layers.dense(net, 1, activation=tf.tanh)
loss = tf.losses.mean_squared_error(prediction, y) 

# pass the second value from iter.get_net() as label
train_op = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(EPOCHS):
        _, loss_value = sess.run([train_op, loss])
        print("Iter: {}, Loss: {:.4f}".format(i, loss_value))