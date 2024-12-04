from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import tensorflow_federated as tff

from utils_tff import *

# loading

# dataset_path = "/home/coder/Desktop/code/dataset_2/trainingSet"
# dataset_path = "/content/drive/MyDrive/dataset_2/trainingSet"
dataset_path = "/content/drive/MyDrive/dataset/trainingSet"
image_paths = list(paths.list_images(dataset_path))

data, labels = load(image_paths, verbose=1)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# processing
dataset_shards = create_shards(data, labels, 10)

train_set_tnr, test_set_tnr = [], []
for client in dataset_shards:
    training , testing = train_test_split(client, test_size=0.1, random_state=42)
    train_set_tnr.append(batch_set(training,32))
    test_set_tnr.append(batch_set(testing,32))


# x_train, x_test ,y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

# train_set_shards = create_shards(x_train,y_train, 10)
# test_set_shards = create_shards(x_test,y_test, 10)

# train_set_tnr =[]
# for train_shard in train_set_shards:
#     train_set_tnr.append(batch_set(train_shard,32))

# train_set_tnr = [batch_set(train_shard, 32) for train_shard in train_set_shards]
# test_set_tnr = [batch_set(test_shard, 32) for test_shard in test_set_shards]

# train_set_tnr = tf.data.Dataset.from_tensor_slices((list(x_train),list(y_train))).shuffle(len(y_train)).batch(len(y_train))
# test_set_tnr = tf.data.Dataset.from_tensor_slices((list(x_test),list(y_test))).shuffle(len(y_test)).batch(len(y_test))


# training

print("[INFO] Aggreagtion")
trainer = tff.learning.algorithms.build_weighted_fed_avg(
    model_fn,
    client_optimizer_fn = tff.learning.optimizers.build_sgdm(learning_rate=0.01, momentum=0.9),
    server_optimizer_fn = tff.learning.optimizers.build_sgdm(learning_rate=0.01, momentum=0.9)
    # client_optimizer_fn = lambda: Adam(),
    # server_optimizer_fn = lambda: Adam()
    )

print("[INFO] Training")
state = trainer.initialize()
for i in range(100):
    state, metrics = trainer.next(state, train_set_tnr)
    print(f"comm_round: {i+1} | acc: {metrics['client_work']['train']['accuracy']} | loss: {metrics['client_work']['train']['loss']}")

# testing

print("[INFO] Evaluation")
evaluator = tff.learning.algorithms.build_fed_eval(model_fn)
state_evl = evaluator.initialize()
for i in range(10):
    metrics = evaluator.next(state_evl, test_set_tnr)
    print(f"comm_round: {i+1} | acc: {metrics}")