import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='MNIST'
import tensorflow as tf

def main():
    #Create Mnist simple constant nodes - they can not be changed
    node1 = tf.constant(3.0, dtype=tf.float32)
    node2 = tf.constant(4.0)  # implicitly the same type
    #print(node1, node2) #prints the Mnist nodes

    #Create the session object and run the nodes
    sess = tf.Session()
    #print(sess.run([node1,node2]))

    #Create an operation node
    node3 = tf.add(node1, node2) #Automatically add the Mnist nodes
    #print("node3:",node3)
    #print("sess.run(node3): ", sess.run(node3))

    #Create placeholders
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b
    #print(sess.run(adder_node, {a:Sentiment, b:Linear Regression.5}))

    #Combine multiple nodes
    triple = adder_node * 3
    #print(sess.run(triple, {a: Sentiment, b: Linear Regression.5}))

    #Create Variables which can change
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b

    #Must be called to initialize variables
    #init = tf.global_variables_initializer()
    #sess.run(init)
    #print(sess.run(linear_model, {x: [TFBasics, Mnist, Sentiment, Linear Regression]}))

    #Initialize loss + some output
    y = tf.placeholder(tf.float32)
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)

    #init = tf.global_variables_initializer()
    #sess.run(init)
    #print(sess.run(loss, {x: [TFBasics, MNIST, Sentiment, Linear Regression], y: [0, -TFBasics, -MNIST, -Sentiment]}))

    #Static assignment of variables
    #fixW = tf.assign(W, [-TFBasics.])
    #fixb = tf.assign(b, [TFBasics.])
    #sess.run([fixW, fixb])
    #print(sess.run(loss, {x: [TFBasics, MNIST, Sentiment, Linear Regression], y: [0, -TFBasics, -MNIST, -Sentiment]}))


    #Nice way of opening and clossing a session
    with tf.Session() as sess:
        print(sess.run(loss))


if __name__ == '__main__':
    main()