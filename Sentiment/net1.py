import tensorflow as tf
import pickle
import numpy as np


pickle_in = open('sentiment.pickle', 'rb')
train_x, train_y, test_x, test_y = pickle.load(pickle_in)

# Number of nodes in the hidden layer
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# Number of classes
n_classes = 2

# Number of pictures in a batch
batch_size = 100

x = tf.placeholder(tf.float32, [None, len(train_x[0])])
y = tf.placeholder(tf.float32)

def network(data):

    #All weights and biases for each Layer in one Tensor
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer =   {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}

    #Connect the layers together
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    l_output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

    return l_output

def train(x):
    prediction = network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
    epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _,c = sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})
                epoch_loss+=c
                i+=batch_size
            print('Epoch: ',epoch,' loss: ',epoch_loss)
            epoch_loss=0

        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ',accuracy.eval({x:test_x, y:test_y}))


train(x)
