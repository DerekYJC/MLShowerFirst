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

testing_set = train_set[219000:]
train_set = train_set[:219000]

# build net
input_x = tf.placeholder(dtype=tf.float32, shape=[None, 96, 96, 3])
input_y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
input_x_nor = input_x/128-1

input_x_ = tf.keras.layers.Input(input_x.shape[1:])

conv1 = tf.keras.layers.Conv2D(
    filters=50,
    kernel_size=(3, 3),
    padding='same',
    activation=tf.nn.relu)(input_x_)
p1 = tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2))(conv1)

conv2 = tf.keras.layers.Conv2D(
    filters=100,
    kernel_size=(3, 3),
    padding='same',
    activation=tf.nn.relu)(p1)
p2 = tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2))(conv2)

conv3 = tf.keras.layers.Conv2D(
    filters=100,
    kernel_size=(3, 3),
    padding='same',
    activation=tf.nn.relu)(p2)
p3 = tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2))(conv3)

f1 = tf.keras.layers.Flatten()(p3)
d1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)(f1)

y_ = tf.keras.layers.Dense(units=1, activation='sigmoid')(d1)

md = tf.keras.Model(inputs=input_x_, outputs=y_)
y_pred = md(input_x_nor)

loss = tf.losses.mean_squared_error(labels=input_y, predictions=y_pred)
opt = tf.train.AdamOptimizer(learning_rate=5e-4)
train = opt.minimize(loss, var_list=md.trainable_variables)

err = tf.math.abs(y_pred-input_y)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# build net


def load_data(zip_path, _df):

    X = np.zeros([0, 96, 96, 3])
    with ZipFile(zip_path, 'r') as zp:
        for i in _df.index:
            with zp.open(_df.infol[i]) as imagefile:
                img = Image.open(imagefile)
                # img.show()
                X = np.concatenate((X, [np.array(img)]), axis=0)

    return X


def training(x_, y_):
    t, l = sess.run([train, loss], feed_dict={input_x: x_, input_y: y_})
    print(l)


def acc(x_, y_):
    er = sess.run(err, feed_dict={input_x: x_, input_y: y_})
    result = (np.where(er < 0.5, 1., 0.))
    acc = np.mean(result)
    return acc


batch_n = int(len(train_set)/batch_size)

# for test
xtb = load_data(train_path, testing_set)
ytb = np.array(testing_set.label.values.reshape([-1, 1]))

# training
for epo in range(10):
    print('in epo:', epo)
    for i in range(batch_n):
        print(i, '/', batch_n)

        xx = train_set[i*batch_size:(i+1)*batch_size]
        x_batch = load_data(train_path, xx)
        y_batch = np.array(
            train_set.label[i*batch_size:(i+1)*batch_size]).reshape([-1, 1])
        training(x_batch, y_batch)

        if i % 10 == 0:
            print('acc:', acc(xtb, ytb))
