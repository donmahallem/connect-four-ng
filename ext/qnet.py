
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, MaxPool2D, Flatten, Embedding, Dot, Input, Lambda, Dropout, BatchNormalization, Conv2D, Activation, Add
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf


def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond = tf.keras.backend.abs(error) < clip_delta

    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

    return tf.where(cond, squared_loss, linear_loss)


'''
 ' Same as above but returns the mean loss.
'''


def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
    return tf.keras.backend.mean(huber_loss(y_true, y_pred, clip_delta))


def res_block(x, filters, kernel, activation="relu"):
    x_in = x
    F1, F2, F3 = filters
    x1 = Conv2D(F1, kernel_size=(1, 1), strides=(1, 1), padding="valid")(x)
    x1 = BatchNormalization(axis=3)(x1)
    x1 = Activation(activation)(x1)
    x1 = Conv2D(F2, kernel_size=kernel, strides=(1, 1), padding="same")(x1)
    x1 = BatchNormalization(axis=3)(x1)
    x1 = Activation(activation)(x1)
    x1 = Conv2D(F3, kernel_size=(1, 1), strides=(1, 1), padding="valid")(x1)
    x1 = BatchNormalization(axis=3)(x1)
    x1 = Activation(activation)(x1)
    out_x = Add()([x1, x_in])
    out_x = Activation(activation)(out_x)
    return out_x


class QNet:
    def __init__(self):
        inputs = Input(shape=(7, 7, 1))
        #x = MaxPool2D(2)(x)
        x = res_block(inputs, (32, 32, 32), 4, activation="linear")
        x = Conv2D(64, (2, 2), padding="valid")(x)
        x = res_block(x, (64, 64, 64), 2)
        x = Conv2D(128, (2, 2), padding="valid")(x)
        x = res_block(x, (128, 128, 128), 2)
        x = Conv2D(256, (2, 2), padding="valid")(x)
        x = res_block(x, (256, 256, 256), 2)
        x = Conv2D(256, (2, 2), padding="valid")(x)
        x = res_block(x, (256, 256, 256), 2)
        x = Conv2D(256, (2, 2), padding="valid")(x)
        x = res_block(x, (256, 256, 256), 2)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(7, activation='linear')(x)

        self._model = Model(inputs=inputs, outputs=x)
        self._model.compile('rmsprop', tf.losses.huber_loss)
        self._model.summary()

    def getModel(self):
        return self._model

    def getWeights(self):
        return self._model.get_weights()

    def setWeights(self, w):
        self._model.set_weights(w)

    def fit(self, data, data_y, epochs=1):
        train_data = data.reshape(
            (data.shape[0], data.shape[1], data.shape[2], 1))
        self._model.fit(train_data, data_y, epochs=epochs, batch_size=1024)

    def predict(self, data):
        train_data = data.reshape(
            (data.shape[0], data.shape[1], data.shape[2], 1))
        return self._model.predict(train_data)

    def save_weights(self, name):
        self._model.save_weights(name)

    def load_weights(self, name):
        self._model.load_weights(name)

    def save(self, name):
        self._model.save(name)


net = QNet()
net.save("model.h5")
