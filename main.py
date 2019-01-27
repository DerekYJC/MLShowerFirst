import tensorflow as tf
import numpy as np
import pandas as pd
from zipfile import ZipFile
from PIL import Image

batch_size = 100


test_path = 'data/test.zip'
train_path = 'data/train.zip'

# get training data info

with ZipFile(train_path, 'r') as trainf:
    train_set = pd.DataFrame({'infol': trainf.infolist()})
train_set['id'] = train_set.infol.map(lambda x: x.filename.split('.')[0])
labels = pd.read_csv('data/train_labels.csv')
train_set = train_set.merge(labels, on='id')

with ZipFile(test_path, 'r') as testf:
    test_set = pd.DataFrame({'infol': testf.infolist()})
test_set['id'] = test_set.infol.map(lambda x: x.filename.split('.')[0])


def load_data(zip_path, _df):

    X = np.zeros([0, 96, 96, 3])
    with ZipFile(zip_path, 'r') as zp:
        for i in _df.index:
            with zp.open(_df.infol[i]) as imagefile:
                img = Image.open(imagefile)
                # img.show()
                X = np.concatenate((X, [np.array(img)]), axis=0)

    return X


input_x = tf.placeholder(dtype=tf.float32, shape=[None, 96, 96, 3])
input_x = input_x/128-1
input_y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

conv1 = tf.layers.conv2d(
    inputs=input_x,
    filters=50,
    kernel_size=(3, 3),
    padding='same',
    activation=tf.nn.relu)
p1 = tf.layers.max_pooling2d(
    inputs=conv1,
    pool_size=(2, 2),
    strides=(2, 2))

conv2 = tf.layers.conv2d(
    inputs=p1,
    filters=100,
    kernel_size=(3, 3),
    padding='same',
    activation=tf.nn.relu)
p2 = tf.layers.max_pooling2d(
    inputs=conv2,
    pool_size=(2, 2),
    strides=(2, 2))

conv3 = tf.layers.conv2d(
    inputs=p2,
    filters=100,
    kernel_size=(3, 3),
    padding='same',
    activation=tf.nn.relu)
p3 = tf.layers.max_pooling2d(
    inputs=conv3,
    pool_size=(2, 2),
    strides=(2, 2))

f1 = tf.layers.flatten(inputs=p3)
d1 = tf.layers.dense(inputs=f1, units=100, activation=tf.nn.relu)

y = tf.layers.dense(inputs=d1, units=1, activation=tf.nn.sigmoid)
loss = tf.losses.mean_squared_error(labels=input_y, predictions=y)
opt = tf.train.AdamOptimizer()
train = opt.minimize(loss)


sess = tf.Session()
sess.run(tf.global_variables_initializer())


def training(x_, y_):
    t, l = sess.run([train, loss], feed_dict={input_x: x_, input_y: y_})
    print(l)


batch_n = int(len(train_set)/batch_size)

for epo in range(10):
    print('in epo:', epo)
    for i in range(batch_n):

        xx = train_set[i*batch_size:(i+1)*batch_size]
        x_batch = load_data(
            train_path, xx)
        y_batch = np.array(
            train_set.label[i*batch_size:(i+1)*batch_size]).reshape([-1, 1])
        training(x_batch, y_batch)
