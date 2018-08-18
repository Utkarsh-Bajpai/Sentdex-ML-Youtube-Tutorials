import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import matplotlib.pyplot as plt

TRAIN_DIR = 'images/train/'
TEST_DIR = 'images/test1/'
IMG_SIZE = 50
learning_rate = 1e-3

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(learning_rate,'1.0')

def label_img(img):
    #Get the label of the image whether cat or dog
    word_label = img.split('.')[-3]
    if word_label == 'cat': return [1,0]
    elif word_label == 'dog' : return [0,1]

def process_train_data():
    train_data = []
    for image in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(image)
        path = os.path.join(TRAIN_DIR, image)
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
        train_data.append([np.array(img), np.array(label)])
    shuffle(train_data)
    np.save('train_data.npy', train_data)
    return train_data


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

#train_data = process_train_data()
#If the file exists
train_data = np.load('train_data.npy')

#test_data = process_test_data()
#If the file exists
test_data = np.load('test_data.npy')

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print(MODEL_NAME, " loaded!")

train = train_data[:-500]
test = train_data[-500:]

#[data,label]
X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = np.array([i[1] for i in train])

test_X = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_Y = np.array([i[1] for i in test])

#model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_X}, {'targets': test_Y}),
#    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

#So we can save the model
model.save(MODEL_NAME)

fig = plt.figure()
for num,data in enumerate(test_data[:12]):
    # Cat = [1,0]
    # Dog = [0,1
    img_num = data[1]
    img_data = data[0]
    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    model_out = model.predict([data])[0]
    if np.argmax(model_out) == 1: str_label = 'Dog'
    else: str_label = 'Cat'

    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()