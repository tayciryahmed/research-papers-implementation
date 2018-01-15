from keras.layers import Conv2D, MaxPooling2D, Input, Dropout, BatchNormalization, Dense, Flatten
from keras.models import Model
import numpy as np
import utils

class ATDA(object):
    def __init__(self, name='mnist-mnistm'):
        pass

    def intial_training(self, inp, F1, F2, source_train, y_train):
        model = Model([inp], [F1, F2])
        model.compile(loss=['mse', 'mse'], optimizer='rmsprop', metrics=['accuracy'])
        model.fit([source_train], [y_train, y_train], nb_epoch=1, batch_size=1)
        return model

    def pseudo_labeling(self, output1, output2, data, true_label, threshold=1e-5):
        id = np.equal(np.argmax(output1,1),np.argmax(output2,1))
        output1 = output1[id,:]
        output2 = output2[id, :]
        data = data[id, :]
        true_label = true_label[id, :]
        max1 = np.max(output1,1)
        max2 = np.max(output2,1)
        id2 = np.max(np.vstack((max1,max2)),0)>threshold
        output1 = output1[id2,:]
        data = data[id2, :]
        pseudo_label = utils.dense_to_one_hot(np.argmax(output1,1),10)
        true_label = true_label[id2, :]
        print data.shape, pseudo_label.shape
        return data, pseudo_label

    def second_training(self, inp, source_train, y_train, target_data, y_target_pseudo_labels, F1, F2, Ft):
        print source_train.shape, target_data.shape, y_train.shape, y_target_pseudo_labels.shape
        data = np.concatenate((source_train, target_data))
        y = np.concatenate((y_train, y_target_pseudo_labels))

        F1F2 = Model([inp], [F1, F2])
        F1F2.compile(loss=['mse', 'mse'], optimizer='rmsprop', metrics=['accuracy'])
        F1F2.fit([data], [y, y], nb_epoch=1, batch_size=1)

        Ft = Model([inp], [Ft])
        Ft.compile(loss=['mse'], optimizer='rmsprop', metrics=['accuracy'])
        Ft.fit([target_data], [y_target_pseudo_labels], nb_epoch=1, batch_size=1)
        return F1F2, Ft

    def iterate_algorithm(self, inp, F1, F2, Ft, source_train, y_train, target_data, target_label, iter=1, k=1):
        for i in range(iter):
            model = self.intial_training(inp, F1, F2, source_train, y_train)

        output1, output2 = model.predict([target_data])

        data, pseudo_label = self.pseudo_labeling(output1, output2, target_data, target_label)

        for i in range(k):
            for j in range(iter):
                F1F2, Ft = self.second_training(inp, source_train, y_train, data, pseudo_label, F1, F2, Ft)
                output1, output2 = F1F2.predict([target_data])
                data, pseudo_label = self.pseudo_labeling(output1, output2, target_data, target_label)

        return Ft

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
        Ft = self.Ft(F)

        Ft = self.iterate_algorithm(inp, F1, F2, Ft, source_train, y_train, target_data, target_label)
        scores = Ft.evaluate(target_data, [target_label])

        print '\nevaluate result: mse={}, accuracy={}'.format(*scores)


        """
        # test model
        model = Model([inp], [F1, F2, F3])
        model.compile(loss=['mse', 'binary_crossentropy', 'categorical_crossentropy'], optimizer='rmsprop', metrics=['accuracy'])

        model.fit([source_train], [y_train, y_train, y_train], nb_epoch=1, batch_size=1)

        scores = model.evaluate(source_train, [y_train, y_train, y_train])

        print '\nevaluate result: mse={}, binary_crossentropy={}, categorical_crossentropy={}, accuracy={}'.format(*scores)
        """
