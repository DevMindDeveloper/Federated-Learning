from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import backend

from utils import *
from model import *

# dataset
dataset_path = "/home/coder/Desktop/code/dataset/trainingSet"
# dataset_path = "/content/drive/MyDrive/dataset/trainingSet"
image_path_list = list(paths.list_images(dataset_path))

data , labels = load(image_path_list, verbose = 1)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# spliting data
x_train, x_test, y_train, y_test = train_test_split(data, labels,
                                                    test_size = 0.1, random_state = 42) 

num_clients = 10
clients = create_clients(x_train, y_train, num_clients, "client")

clients_batchset = {}
for (name,shard) in clients.items():
    clients_batchset[name] = batch_data(shard)

test_batchset = tf.data.Dataset.from_tensor_slices((x_test,y_test)).shuffle(len(y_test)).batch(len(y_test))


## security & training

# training initialization
mlp_model = SimpleMLP()
global_model = mlp_model.build(784,10)
global_model_weight = []

# security initialization
dim = len(tensor_list(weight_tenosr(global_model.get_weights())))

# ------------------------------------------------------------------
# print("Length",len(weight_tenosr(global_model.get_weights())))
# print(global_model.summary())
# for layer in range(0,len(global_model.layers),2):
#     w, b = global_model.layers[layer].get_weights()
#     print(w.shape)
# ------------------------------------------------------------------

s = MVNP(num_clients)
np.random.seed(s)
r = np.random.randint(100,size=dim)
r = tf.cast(tf.convert_to_tensor(r), dtype = tf.float32)
# r = tf.convert_to_tensor(r)

for comm_round in range(comms_round):

    gm_weights = global_model.get_weights()
    scaled_weigth_list=[]
    client_masked_weight = []

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

        # ---------------------------------------------------------------
        # for i,w in enumerate(scaled_weight):
        #     print(f"shape of scaled weight_{i}",np.array(w).shape)
        # ---------------------------------------------------------------

        tensor_weight = weight_tenosr(scaled_weight)
        masked_tensor = to_mask(tensor_weight, r)
        client_masked_weight = masked_tensor[None,:] if len(client_masked_weight) == 0 else tf.concat([client_masked_weight, masked_tensor[None,:]], axis=0)

        backend.clear_session()
    
    for weight in client_masked_weight:
        global_model_weight = tf.expand_dims(weight, axis=0) if len(global_model_weight) == 0 else tf.concat([global_model_weight, tf.expand_dims(weight, axis = 0)], axis=0)
    
    print("before agg")
    average_weight = weight_aggregation(global_model_weight)
    print("after agg")

    global_model_weight_unmasked = to_mask(average_weight, 0-r)

    # ---------------------------------------------------------------
    print("for setting",global_model_weight_unmasked.shape)
    # ---------------------------------------------------------------

    layer1_weights = tf.reshape(global_model_weight_unmasked[:156800], (784, 200))
    layer1_biases = global_model_weight_unmasked[156800:157000]
    layer2_weights = tf.reshape(global_model_weight_unmasked[157000:197000], (200, 200))
    layer2_biases = global_model_weight_unmasked[197000:197200]
    layer3_weights = tf.reshape(global_model_weight_unmasked[197200:199200], (200, 10))
    layer3_biases = global_model_weight_unmasked[199200:]

    weights = [
        [layer1_weights, layer1_biases],
        [layer2_weights, layer2_biases],
        [layer3_weights, layer3_biases],
    ]

    print("before set")
    global_model.set_weights([item for sublist in weights for item in sublist])
    print("after set")

    for (x_test, y_test) in test_batchset:
        global_acc, global_loss = test_model(x_test, y_test, global_model, comm_round)














# print(clients_batchset['client_1'])
# print(list(clients_batchset['client_1'])[0])
# print(list(clients_batchset['client_1'])[0][0])
# print(list(clients_batchset['client_1'])[0][0].shape[0])
# print(tf.data.experimental.cardinality(clients_batchset['client_1']))