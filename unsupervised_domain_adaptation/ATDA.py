from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Conv2D, MaxPooling2D, Embedding, Merge, Dropout, BatchNormalization
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

import numpy as np
import random
import utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM

class ATDA(object):
    def __init__(self, name='mnist-mnistm'):
        pass

    """
    def intial_training(self, inp, F1, F2, source_train, y_train):
        model = Model([inp], [F1, F2])
        model.compile(loss=['mse', 'mse'], optimizer='rmsprop', metrics=['accuracy'])
        model.fit([source_train], [y_train, y_train, nb_epoch=1, batch_size=1)
        return model

    def pseudo_labeling(self, model, target_data):
        return y_target_pseudo_labels

    def second_training(self, inp, source_train, y_train, target_data, y_target_pseudo_labels, F1, f2, Ft):
        data = np.concatenate(source_train, target_data)
        y = np.concatenate(y_train, y_target_pseudo_labels)

        F1F2 = Model([inp], [F1, F2])
        F1F2.compile(loss=['mse', 'mse'], optimizer='rmsprop', metrics=['accuracy'])
        F1F2.fit([data], [y, y], nb_epoch=1, batch_size=1)

        Ft = Model([inp], [Ft])
        Ft.compile(loss=['mse', 'mse'], optimizer='rmsprop', metrics=['accuracy'])
        Ft.fit([target_data], [y_target_pseudo_labels], nb_epoch=1, batch_size=1)
        return F1f2, Ft

    def iterative(self, F1, F2, F3, source_train, y_train, target_data, target_label, iter):
        for i in range(iter):


    """


    def shared_network(self):
        inp = Input(shape=(28,28,3))
        x = Conv2D(kernel_size=(5,5), filters=32, strides=(1, 1), activation="relu", padding="same")(inp)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(x)
        x = Conv2D(kernel_size=(5,5), filters=48, strides=(1, 1), activation="relu", padding="same")(inp)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(x)
        x = Flatten()(x)
        return x, inp

    def F1(self, f):
        x = BatchNormalization()(f)
        x = Dense(100, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = Dense(100, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = Dense(10, activation="softmax")(x)
        return x

    def F2(self, f):
        x = BatchNormalization()(f)
        x = Dense(100, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = Dense(100, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = Dense(10, activation="softmax")(x)
        return x

    def Ft(self, f):
        x = BatchNormalization()(f)
        x = Dense(100, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = Dense(100, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = Dense(10, activation="softmax")(x)
        return x

    def fit_ATDA(self, source_train, y_train, target_val, y_val, target_data, \
                    target_label, nb_epoch=5, k_epoch=100, batch_size=128, \
                    shuffle=True,N_init=5000,N_max=40000):

        F, inp = self.shared_network()
        F1 = self.F1(F)
        F2 = self.F2(F)
        F3 = self.Ft(F)

        print F1.shape, F2.shape, F3.shape
        model = Model([inp], [F1, F2, F3])
        model.compile(loss=['mse', 'binary_crossentropy', 'categorical_crossentropy'], optimizer='rmsprop', metrics=['accuracy'])

        model.fit([source_train], [y_train, y_train, y_train], nb_epoch=1, batch_size=1)

        scores = model.evaluate(source_train, [y_train, y_train, y_train], verbose=0)

        print '\nevaluate result: mse={}, binary_crossentropy={}, categorical_crossentropy={}, accuracy={}'.format(*scores)
