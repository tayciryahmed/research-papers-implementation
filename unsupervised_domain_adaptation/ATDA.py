from keras.layers import Conv2D, MaxPooling2D, Input, Dropout, BatchNormalization, Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import utils

class ATDA(object):
    def __init__(self, name='mnist-mnistm'):
        pass

    def intial_training(self, inp, F1, F2, Ft, X_source_train, y_source_train, n_epoch, batch_size_F1F2, batch_size_Ft, lr):
        modelF1F2 = Model([inp], [F1, F2])
        modelF1F2.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'], optimizer='adam', metrics=['accuracy'])
        modelF1F2.fit([X_source_train], [y_source_train, y_source_train], nb_epoch=n_epoch, batch_size=batch_size_F1F2)

        modelFt = Model([inp], [F1, F2])
        modelFt.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'], optimizer='adam', metrics=['accuracy'])
        modelFt.fit([X_source_train], [y_source_train, y_source_train], nb_epoch=n_epoch, batch_size=batch_size_Ft)

        return modelF1F2, modelFt

    def pseudo_labeling(self, output1, output2, X_target_train, threshold):
        id = np.equal(np.argmax(output1,1),np.argmax(output2,1))
        output1 = output1[id,:]
        output2 = output2[id, :]
        data_pseudo_labeled = X_target_train[id, :]
        max1 = np.max(output1,1)
        max2 = np.max(output2,1)
        id2 = np.max(np.vstack((max1,max2)),0)>threshold
        output1 = output1[id2,:]
        data_pseudo_labeled = data_pseudo_labeled[id2, :]
        pseudo_label = utils.dense_to_one_hot(np.argmax(output1,1),10)
        print data_pseudo_labeled.shape, pseudo_label.shape
        return data_pseudo_labeled, pseudo_label

    def second_training(self, inp, X_source_train, y_source_train, X_target_train, y_target_pseudo_labels, F1, F2, Ft, n_epoch, batch_size_F1F2, batch_size_Ft, lr):
        print X_source_train.shape, X_target_train.shape, y_source_train.shape, y_target_pseudo_labels.shape
        data = np.concatenate((X_source_train, X_target_train))
        y = np.concatenate((y_source_train, y_target_pseudo_labels))

        F1F2 = Model([inp], [F1, F2])
        opt = Adam(lr=0.01)
        F1F2.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'], optimizer=opt, metrics=['accuracy'])
        F1F2.fit([data], [y, y], nb_epoch=n_epoch, batch_size=batch_size_F1F2)

        Ft_model = Model([inp], [Ft])
        Ft_model.compile(loss=['categorical_crossentropy'], optimizer='adam', metrics=['accuracy'])
        Ft_model.fit([X_target_train], [y_target_pseudo_labels], nb_epoch=n_epoch, batch_size=batch_size_Ft)
        return F1F2, Ft_model

    def iterate_algorithm(self, inp, F1, F2, Ft, X_source_train, y_source_train, X_target_train, y_target_train, n_epoch, k, batch_size_F1F2, batch_size_Ft, threshold, lr):

        modelF1F2, modelFt = self.intial_training(inp, F1, F2, Ft, X_source_train, y_source_train, n_epoch, batch_size_F1F2, batch_size_Ft, lr)

        output1, output2 = modelF1F2.predict([X_target_train])

        data_pseudo_labeled, pseudo_label = self.pseudo_labeling(output1, output2, X_target_train, threshold)

        for i in range(k):
            F1F2, Ft_model = self.second_training(inp, X_source_train, y_source_train, data_pseudo_labeled, pseudo_label, F1, F2, Ft, n_epoch, batch_size_F1F2, batch_size_Ft, lr)
            output1, output2 = F1F2.predict([X_target_train])
            data_pseudo_labeled, pseudo_label = self.pseudo_labeling(output1, output2, X_target_train, threshold)

        return Ft_model

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
        x = Dropout(0.5)(x)
        x = Dense(100, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(10, activation="softmax")(x)
        return x

    def F2(self, f):
        x = BatchNormalization()(f)
        x = Dense(100, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(100, activation="relu")(x)
        x = Dropout(0.5)(x)
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


    def fit_ATDA(self, X_source_train, y_source_train, X_target_test,\
                        y_target_test, X_target_train, y_target_train, \
                        threshold, n_epoch, k, batch_size_F1F2, batch_size_Ft, lr):

        F, inp = self.shared_network()
        F1 = self.F1(F)
        F2 = self.F2(F)
        Ft = self.Ft(F)

        Ft = self.iterate_algorithm(inp, F1, F2, Ft, X_source_train, y_source_train, X_target_train, y_target_train, n_epoch, k, batch_size_F1F2, batch_size_Ft, threshold, lr)
        scores = Ft.evaluate(X_target_test, [y_target_test])

        print '\nevaluate result: categorical_crossentropy={}, accuracy={}'.format(*scores)















        """
        # test model
        model = Model([inp], [F1, F2, F3])
        model.compile(loss=['categorical_crossentropy', 'binary_crossentropy', 'categorical_crossentropy'], optimizer='adam', metrics=['accuracy'])

        model.fit([X_source_train], [y_source_train, y_source_train, y_source_train], nb_epoch=1, batch_size=1)

        scores = model.evaluate(X_source_train, [y_source_train, y_source_train, y_source_train])

        print '\nevaluate result: categorical_crossentropy={}, binary_crossentropy={}, categorical_crossentropy={}, accuracy={}'.format(*scores)
        """
