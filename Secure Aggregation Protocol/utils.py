import cv2
import numpy as np
import os
import random
import tensorflow as tf
from sklearn.metrics import accuracy_score 
from phe import paillier

def load(paths, verbose = -1):
    print("[INFO] loading dataset!")

    data = []
    label_list = []

    for (i, impath) in enumerate(paths):
        image_gray = cv2.imread(impath, cv2.IMREAD_GRAYSCALE)
        image = np.array(image_gray).flatten()

        label = impath.split(os.path.sep)[-2]

        data.append(image/255)
        label_list.append(label)

        if verbose > 0:
            print(f"[INFO] processed {i+1}/{len(paths)}")
    
    return data, label_list

def create_clients(images, labels, no_clients = 10, initials = "clients"):
    print("[INFO] creating client's shards!")

    clients = [f"{initials}_{i+1}" for i in range(no_clients)]

    data = list(zip(images, labels))
    random.shuffle(data)

    size = len(data)//no_clients
    shards = [data[i:i+size] for i in range(0, size*no_clients, size)]

    return {clients[i]:shards[i] for i in range(no_clients)}

def batch_data(data, bs=32):
    # print("[INOF] creating batch set!")

    images, labels = zip(*data)
    dataset = tf.data.Dataset.from_tensor_slices((list(images), list(labels)))
    return dataset.shuffle(len(images)).batch(bs)

def weight_scaling_factor(clients_tnr_batchset, client):
    # print("[INFO] creating factor for scaling!")

    client_names = list(clients_tnr_batchset.keys())
    bs = list(clients_tnr_batchset[client])[0][0].shape[0]
    global_count = sum(tf.data.experimental.cardinality(clients_tnr_batchset[client]).numpy() for client in client_names)*bs
    local_count = (tf.data.experimental.cardinality(clients_tnr_batchset[client]).numpy())*bs
    return local_count/global_count

def sacling_weight(weight, scalar):
    # print("[INFO] scaling weight!")

    weight_list = []
    for i in range(len(weight)):
        weight_list.append(scalar * weight[i])
    return weight_list

def weight_aggregation(scaled_weight_list):
    # print("[INFO] aggregatation process!")

    avg_weigth = []
    for grad_tuple_list in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_tuple_list, axis=0)
        avg_weigth.append(layer_mean)
    return avg_weigth

def test_model(x_test, y_test, model, comm_round):
    # print("[INFO] testing model!")

    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    y_pred = model.predict(x_test)
    loss = cce(y_test, y_pred)
    acc = accuracy_score(tf.argmax(y_pred, axis=1), tf.argmax(y_test, axis=1))
    print(f"com_rou: {comm_round} | accuracy: {acc:.3%} | loss: {loss}")
    return acc, loss

# security

def MVNP(n):
    print("[INFO] Creating Seed!")

    public_key, private_key = paillier.generate_paillier_keypair()
    enc_r = []

    for i in range(n):
        enc_r.append([public_key.encrypt(x) for x in np.random.randint(100,size=1).tolist()])
    enc_s = np.sum(enc_r, axis=0)
    R = [int(private_key.decrypt(x)) for x in enc_s]
    return R

def to_mask(w,r):
    print("[INFO] Appling Mask!")

    value = w+r
    return value

def weight_tenosr(w):
    print("[INFO] Combining Tensor!")

    value = []
    # for key in w.keys():
    for key in range(len(w)):
        value = tf.reshape(w[key],[-1]) if not len(value) else tf.concat([value, tf.reshape(w[key],[-1])],axis=0)
    return value

def tensor_list(w):
    value = w.numpy().tolist()
    return value

    