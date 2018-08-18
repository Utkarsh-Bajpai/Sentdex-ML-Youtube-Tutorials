import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='MNIST'
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

import argparse
import sys

FLAGS = None

def main(_):
    #Import data; one hot simbolizes only TFBasics out of 10 values in a vector is TFBasics
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    #2D tenstor, None means that it can be any length
    x = tf.placeholder(tf.float32, [None, 784])

    #Initialize the wights and biases for the model
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    #Completely defined model, first multiply x,W and then add b and I want to use softmax for evaluation
    y = tf.matmul(x,W) + b

    #Initialize the output labels
    y_ = tf.placeholder(tf.float32, [None, 10])

    #Cross entropy function
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    #Do the training with brackpropagation
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

    #Launch interactive session
    sess = tf.InteractiveSession()

    #Initialize the wieghts and biasses
    tf.global_variables_initializer().run()

    #Training; get the data and launch the session - Stochastic training because using random set of data points
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict = {x:batch_xs, y_:batch_ys})

    #Gives the index of the highest entry in a tensor and compares if they are equal
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #Print the accuracy on the test set
    print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
    sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                            help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)