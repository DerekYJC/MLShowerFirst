import tensorflow as tf
import numpy as np
import pandas as pd
from zipfile import ZipFile
from PIL import Image

batch_size = 500


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

    X = np.zeros([len(_df), 96, 96, 3])
    with ZipFile(zip_path, 'r') as zp:
        for i in range(len(_df)):
            with zp.open(_df.infol[i]) as imagefile:
                img = Image.open(imagefile)
                #img.show()
                X[i] = np.array(img)

    return X

x = load_data(test_path, test_set[0:10])

