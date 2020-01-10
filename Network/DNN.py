import numpy as np
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras.utils.generic_utils import get_custom_objects
from tensorflow.python.keras import optimizers, backend as K
from tensorflow.python.keras.callbacks import CSVLogger, Callback

class DNN():
    def __init__(self):
        self.model = None
        get_custom_objects().update({'bent': Bent(bent)})

    def predict(self, x):
        prediction = self.model.predict(x)
        return prediction

    def create_model(self, input_dim):
        self.model = Sequential()
        self.model.add(Dense(units=32, input_dim=input_dim, activation='bent'))
        self.model.add(Dropout(0.05))
        self.model.add(Dense(units=16, activation='bent'))
        self.model.add(Dense(units=8, activation='bent'))
        self.model.add(Dense(units=1, activation='bent'))

    def configure(self):
        optimizer = optimizers.Adam(lr=1e-4)
        self.model.compile(optimizer=optimizer, loss="mean_squared_error")

    def train(self, x, y, valX, valY, batchSize=32, epochs=100):
        # Callbacks
        trainValCb = TrainValCallback()
        csv_logger = CSVLogger('training.log')
        # Training
        self.model.fit(x, y, validation_data=(valX, valY), batch_size=batchSize, epochs=epochs, verbose=2, shuffle=True, callbacks=[trainValCb, csv_logger])


def load_network(suffix="train"):
    dnn = DNN()
    dnn.model = load_model("dnn{}.h5".format(suffix))
    return dnn


def create_training_dataset_with_squence(trainData, outputIdx, lookback):
    # outputIdx is the index of out output column
    dim_0 = trainData.shape[0] - lookback
    dim_1 = trainData.shape[1]
    x = np.zeros((dim_0, lookback, dim_1))
    y = np.zeros((dim_0,))

    for i in range(dim_0):
        x[i] = trainData[i:lookback + i]
        y[i] = trainData[lookback + i, outputIdx]
    print("Shape of Input and Output data ", np.shape(x), np.shape(y))
    return x, y


class TrainValCallback(Callback):
    def __init__(self):
        super().__init__()
        self.distance = np.inf

    def on_epoch_end(self, epoch, logs={}):
        currentDistance = logs.get('loss')**2 + logs.get('val_loss')**2
        if currentDistance < self.distance:
            self.distance = currentDistance
            self.model.save("dnntrainval.h5")


class Bent(Activation):

    def __init__(self, activation, **kwargs):
        super(Bent, self).__init__(activation, **kwargs)
        self.__name__ = 'bent'


def bent(x):
    return ((K.sqrt(K.pow(x, 2) + 1) - 1) / 2) + x