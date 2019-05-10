import tensorflow as tf
from tensorflow import keras
import numpy as np

# feedable iterator to switch between iterators
EPOCHS = 10

# making fake data using numpy
train_data = ( 
    np.random.sample((100,2)), 
    np.random.sample((100,1))
)
test_data = (np.random.sample((10,2)), np.random.sample((10,1)))

print(test_data)


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


# create placeholder
x, y = tf.placeholder(tf.float32, shape=[None,2]), tf.placeholder(tf.float32, shape=[None,1])

# create two datasets, one for training and one for test
train_dataset = tf.data.Dataset.from_tensor_slices((x,y))
test_dataset = tf.data.Dataset.from_tensor_slices((x,y))

# create the iterators from the dataset
train_iterator = train_dataset.make_initializable_iterator()
test_iterator = test_dataset.make_initializable_iterator()

# same as in the doc https://www.tensorflow.org/programmers_guide/datasets#creating_an_iterator
handle = tf.placeholder(tf.string, shape=[])
iter = tf.data.Iterator.from_string_handle(
    handle, train_dataset.output_types, train_dataset.output_shapes)

next_elements = iter.get_next()

with tf.Session() as sess:

    train_handle = sess.run(train_iterator.string_handle())
    test_handle = sess.run(test_iterator.string_handle())
    
    # initialise iterators. 
    sess.run(train_iterator.initializer, feed_dict={ x: train_data[0], y: train_data[1]})
    sess.run(test_iterator.initializer, feed_dict={ x: test_data[0], y: test_data[1]})
    
    for _ in range(EPOCHS):
        x,y = sess.run(next_elements, feed_dict = {handle: train_handle})
        print(x, y)
    print('')
    x,y = sess.run(next_elements, feed_dict = {handle: test_handle})
    print(x,y)