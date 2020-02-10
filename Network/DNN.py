import numpy as np
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Activation
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
        self.model.add(Dense(units=25, input_dim=input_dim, activation='bent'))
        self.model.add(Dense(units=25, activation='bent'))
        self.model.add(Dense(units=15, input_dim=input_dim, activation='bent'))
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
    dnn.model = load_model("Results/dnn{}.h5".format(suffix))
    return dnn


class TrainValCallback(Callback):
    def __init__(self):
        super().__init__()
        self.distance = np.inf

    def on_epoch_end(self, epoch, logs={}):
        currentDistance = logs.get('loss')**2 + logs.get('val_loss')**2
        if currentDistance < self.distance:
            self.distance = currentDistance
            self.model.save("Results/dnntrainval.h5")

class Bent(Activation):

    def __init__(self, activation, **kwargs):
        super(Bent, self).__init__(activation, **kwargs)
        self.__name__ = 'bent'


def bent(x):
    return ((K.sqrt(K.pow(x, 2) + 1) - 1) / 2) + x