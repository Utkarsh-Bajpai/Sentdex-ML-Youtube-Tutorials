import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='MNIST'
import tensorflow as tf
import numpy as np

def LR1():
    sess = tf.Session()
    #Initialize the variables
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    #Initialize the output
    y = tf.placeholder(tf.float32)
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)
    #Launch optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)
    #Initialize the variables to random values
    init = tf.global_variables_initializer()
    sess.run(init)
    #Do the training
    for i in range(1000):
        sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
    #Print weight and bias
    print(sess.run([W,b]))
    #Print final result
    print(sess.run(linear_model, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
    sess.close()

def LR2():
    #Declare the list of features
    features = [tf.contrib.layers.real_valued_column("x",dimension=1)]

    #Linear Regression estimator
    estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

    #Initialize the training data
    x_train = np.array([1., 2., 3., 4.])
    y_train = np.array([0., -1., -2., -3.])
    x_eval = np.array([2., 5., 8., 1.])
    y_eval = np.array([-1.01, -4.1, -7, 0.])
    input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x_train}, y_train,
                                                  batch_size=4,
                                                  num_epochs=1000)
    #Initialize the validation data
    eval_input_fn = tf.contrib.learn.io.numpy_input_fn(
        {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000)

    #1000 epoch Training set
    estimator.fit(input_fn=input_fn, steps=1000)

    #Train & Evaluate
    train_loss = estimator.evaluate(input_fn=input_fn)
    eval_loss = estimator.evaluate(input_fn=eval_input_fn)
    print("train loss: %r" % train_loss)
    print("eval loss: %r" % eval_loss)


def LR3():
    # Declare the list of features
    features = [tf.contrib.layers.real_valued_column("x", dimension=2)]

    # Linear Regression estimator
    estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

    # Initialize the training data
    x_train = np.array([[90,18],[60,9], [50,3], [30,12]])
    y_train = np.array([1, 1, 0, 0])
    x_eval = np.array([[70,10],[60,9], [10,0], [100,20]])
    y_eval = np.array([1, 1, 0, 1])
    input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x_train}, y_train,
                                                  batch_size=4,
                                                  num_epochs=1000)
    # Initialize the validation data
    eval_input_fn = tf.contrib.learn.io.numpy_input_fn(
        {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000)

    # 1000 epoch Training set
    estimator.fit(input_fn=input_fn, steps=1000)

    # Train & Evaluate
    train_loss = estimator.evaluate(input_fn=input_fn)
    eval_loss = estimator.evaluate(input_fn=eval_input_fn)
    print("train loss: %r" % train_loss)
    print("eval loss: %r" % eval_loss)

def main():
    #LR1()
    #LR2()
    LR3()

if __name__ == '__main__':
    main()
