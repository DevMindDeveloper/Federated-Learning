import cv2
import numpy as np
import os

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.losses import CategoricalCrossentropy
from keras.metrics import Accuracy

import tensorflow as tf
import tensorflow_federated as tff



def load(paths, verbose):
    print("[INFO] loading dataset")

    data_list = []
    label_list = []

    for (i,impath) in enumerate(paths):
        image = cv2.imread(impath, cv2.IMREAD_GRAYSCALE)
        image = np.array(image).flatten()
        data_list.append(image/255)

        label = impath.split(os.path.sep)[-2]
        label_list.append(label)

        if verbose:
            print(f"[INFO] processed {i+1}/{len(paths)}")

    return data_list, label_list

def create_shards(image, label, no_clients):
    print("[INFO] dsitributing dataset")

    data = list(zip(image,label))
    size =  len(data)//no_clients
    shards = [data[i:i+size] for i in range(0,size*no_clients, size)]
    return shards

def batch_set(data, bs):
    print("[INFO] batching")

    image, label = zip(*data)
    dataset = tf.data.Dataset.from_tensor_slices((list(image), list(label)))
    dataset = dataset.shuffle(len(label)).batch(bs)
    # print(dataset.element_spec)
    return dataset

def input_specification():
    return (
        tf.TensorSpec([None, 784], tf.float64),
        tf.TensorSpec([None, 10], tf.int64)
    )

def model_fn():

    model= Sequential()
    model.add(Dense(200, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return tff.learning.models.from_keras_model(
        model,
        input_spec= input_specification(),
        loss= CategoricalCrossentropy(from_logits = True),
        metrics= [Accuracy()]
    )