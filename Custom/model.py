from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.optimizers.legacy import SGD

lr = 0.01
comms_round = 25
loss = "categorical_crossentropy"
metrics = ['accuracy']
optimzer = SGD(learning_rate= lr,
               decay = lr / comms_round,
               momentum=0.9)

class SimpleMLP:
    @staticmethod
    def build(shapes,classes):
        model = Sequential()
        model.add(Dense(200,input_shape=(shapes,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model
