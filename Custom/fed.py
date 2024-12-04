from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import backend

from utils import *
from model import *

dataset_path = "/home/coder/Desktop/code/dataset/trainingSet"
image_path_list = list(paths.list_images(dataset_path))

data , labels = load(image_path_list, verbose = 1)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels,
                                                    test_size = 0.1, random_state = 42) 

clients = create_clients(x_train, y_train, 10, "client")

clients_batchset = {}
for (name,shard) in clients.items():
    clients_batchset[name] = batch_data(shard)

test_batchset = tf.data.Dataset.from_tensor_slices((x_test,y_test)).shuffle(len(y_test)).batch(len(y_test))


mlp_model = SimpleMLP()
global_model = mlp_model.build(784,10)

for comm_round in range(comms_round):

    gm_weights = global_model.get_weights()
    scaled_weigth_list=[]

    clients_names = list(clients_batchset.keys())
    random.shuffle(clients_names)

    for client in clients_names:
        # print(f"[INDO] \"{client}\"")

        local_model = mlp_model.build(784,10)
        local_model.compile(optimizer=optimzer,
                            loss=loss,
                            metrics=metrics)
        
        local_model.set_weights(gm_weights)
        local_model.fit(clients_batchset[client], epochs=1,verbose=0)

        lm_weights = local_model.get_weights()
        weight_factor = weight_scaling_factor(clients_batchset, client)
        scaled_weight = sacling_weight(lm_weights, weight_factor)
        scaled_weigth_list.append(scaled_weight)

        backend.clear_session()
    
    average_weight = weight_aggregation(scaled_weigth_list)
    global_model.set_weights(average_weight)

    for (x_test, y_test) in test_batchset:
        global_acc, global_loss = test_model(x_test, y_test, global_model, comm_round)














# print(clients_batchset['client_1'])
# print(list(clients_batchset['client_1'])[0])
# print(list(clients_batchset['client_1'])[0][0])
# print(list(clients_batchset['client_1'])[0][0].shape[0])
# print(tf.data.experimental.cardinality(clients_batchset['client_1']))