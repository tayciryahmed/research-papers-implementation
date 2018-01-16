from keras.layers import Conv2D, MaxPooling2D, Input, Dropout, BatchNormalization, Dense, Flatten
from keras.models import Model
from keras import optimizers
import numpy as np
import utils
from sklearn.utils import resample
from sklearn.model_selection import StratifiedShuffleSplit


class ATDA(object):
    def __init__(self, name='mnist-mnistm'):
        pass

    def shared_network(self):
        inp = Input(shape=(28, 28, 3))
        x = Conv2D(kernel_size=(5, 5), filters=32, strides=(
            1, 1), activation="relu", padding="same")(inp)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = Conv2D(kernel_size=(5, 5), filters=48, strides=(
            1, 1), activation="relu", padding="same")(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        return x, inp

    def F1(self, f):
        x = Dense(100, activation="relu")(f)
        x = Dropout(0.5)(x)
        x = Dense(100, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(10, activation="softmax")(x)
        return x

    def F2(self, f):
        x = Dense(100, activation="relu")(f)
        x = Dropout(0.5)(x)
        x = Dense(100, activation="relu")(x)
        x = Dropout(0.5)(x)
        x = Dense(10, activation="softmax")(x)
        return x

    def Ft(self, f):
        x = Dense(100, activation="relu")(f)
        x = Dropout(0.2)(x)
        x = Dense(100, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = Dense(10, activation="softmax")(x)
        return x

    def pseudo_labeling(self, output1, output2, X_target_train, threshold, Nt):
        id = np.equal(np.argmax(output1, 1), np.argmax(output2, 1))
        output1 = output1[id, :]
        output2 = output2[id, :]
        data_pseudo_labeled = X_target_train[id, :]
        max1 = np.max(output1, 1)
        max2 = np.max(output2, 1)
        id2 = np.max(np.vstack((max1, max2)), 0) > threshold
        output1 = output1[id2, :]
        data_pseudo_labeled = data_pseudo_labeled[id2, :]
        pseudo_label = utils.dense_to_one_hot(np.argmax(output1, 1), 10)
        data_pseudo_labeled, pseudo_label = resample(
        data_pseudo_labeled, pseudo_label, replace=False, n_samples=Nt, random_state=42)
        return data_pseudo_labeled, pseudo_label

    def intial_training(self, inp, F1, F2, Ft, X_source_train, y_source_train,\
                n_epoch, batch_size_F1F2, batch_size_Ft, lr):

        modelF1F2 = Model([inp], [F1, F2])
        opt = optimizers.SGD(lr=lr, momentum=0.9)
        modelF1F2.compile(loss=['categorical_crossentropy',
                                'categorical_crossentropy'], optimizer=opt,
                                metrics=['accuracy'])
        modelF1F2.fit([X_source_train], [y_source_train, y_source_train],
                      epochs=n_epoch, batch_size=batch_size_F1F2, verbose=2,
                      validation_split=0.2)

        modelFt = Model([inp], [Ft])
        opt = optimizers.SGD(lr=lr, momentum=0.9)
        modelFt.compile(loss=['categorical_crossentropy'],
                        optimizer=opt, metrics=['accuracy'])
        modelFt.fit([X_source_train], [y_source_train], epochs=n_epoch,
                    batch_size=batch_size_Ft, verbose=2, validation_split=0.2)

        return modelF1F2, modelFt

    def second_training(self, inp, X_source_train, y_source_train,
            X_target_train, y_target_pseudo_labels, F1, F2, Ft, n_epoch,
            batch_size_F1F2, batch_size_Ft, lr, X_target_valid, y_target_valid):

        data = np.concatenate((X_source_train, X_target_train))
        y = np.concatenate((y_source_train, y_target_pseudo_labels))

        modelF1F2 = Model([inp], [F1, F2])
        opt = optimizers.SGD(lr=lr, momentum=0.9)
        modelF1F2.compile(loss=['categorical_crossentropy',
                                'categorical_crossentropy'],
                                optimizer=opt, metrics=['accuracy'])
        modelF1F2.fit([data], [y, y], epochs=n_epoch,
                      batch_size=batch_size_F1F2, verbose=2,
                      validation_split=0.2)

        Ft_model = Model([inp], [Ft])
        opt = optimizers.SGD(lr=lr, momentum=0.9)
        Ft_model.compile(loss=['categorical_crossentropy'],
                         optimizer=opt, metrics=['accuracy'])

        Ft_model.fit([X_target_train], [y_target_pseudo_labels], epochs=n_epoch,
                     batch_size=batch_size_Ft, verbose=2,
                     validation_data=(X_target_valid, y_target_valid))

        return modelF1F2, Ft_model

    def iterate_algorithm(self, inp, F1, F2, Ft, X_source_train, y_source_train,
            X_target_train, y_target_train, n_epoch, k, batch_size_F1F2,
            batch_size_Ft, threshold, lr, X_target_test, y_target_test,
                          X_target_valid, y_target_valid):

        print "Initial training"
        modelF1F2, modelFt = self.intial_training(
            inp, F1, F2, Ft, X_source_train, y_source_train, n_epoch, \
                    batch_size_F1F2, batch_size_Ft, lr)
        output1, output2 = modelF1F2.predict([X_target_train])
        Nt = 5000
        data_pseudo_labeled, pseudo_label = self.pseudo_labeling(
            output1, output2, X_target_train, threshold, Nt)

        print "second training"

        for i in range(1, k):
            modelF1F2, Ft_model = self.second_training(inp, X_source_train,
                    y_source_train, data_pseudo_labeled, pseudo_label, F1,
                    F2, Ft, n_epoch, batch_size_F1F2, batch_size_Ft, lr,
                    X_target_valid, y_target_valid)

            output1, output2 = modelF1F2.predict([X_target_train])

            Nt = int((float(i) / k) * X_target_train.shape[0])

            if Nt > 40000: break

            data_pseudo_labeled, pseudo_label = self.pseudo_labeling(
                output1, output2, X_target_train, threshold, Nt)

            scores = Ft_model.evaluate(X_target_test, [y_target_test], verbose=2)
            print '\nevaluate result: categorical_crossentropy={}, accuracy={}'.format(*scores)

        return Ft_model

    def fit_ATDA(self, X_source_train, y_source_train, X_target_test,
                 y_target_test, X_target_train_valid, y_target_train_valid,
                 threshold, n_epoch, k, batch_size_F1F2, batch_size_Ft, lr):

        F, inp = self.shared_network()
        F1 = self.F1(F)
        F2 = self.F2(F)
        Ft = self.Ft(F)

        # validation data
        sss = StratifiedShuffleSplit(n_splits=3, test_size=0.018, random_state=42)
        for train_index, test_index in sss.split(X_target_train_valid, y_target_train_valid):
            X_target_train, X_target_valid = X_target_train_valid[train_index,:],\
                        X_target_train_valid[test_index, :]
            y_target_train, y_target_valid = y_target_train_valid[train_index,:], \
                    y_target_train_valid[test_index, :]

        print X_target_valid.shape, y_target_valid.shape, X_target_test.shape,\
            y_target_test.shape, X_source_train.shape, y_source_train.shape, \
            X_target_train.shape, y_target_train.shape

        Ft_model = self.iterate_algorithm(inp, F1, F2, Ft, X_source_train,
             y_source_train, X_target_train, y_target_train, n_epoch, k,
             batch_size_F1F2, batch_size_Ft, threshold, lr, X_target_test,
             y_target_test, X_target_valid, y_target_valid)

        scores = Ft_model.evaluate(X_target_test, [y_target_test], verbose=2)

        print '\nevaluate result: categorical_crossentropy={}, accuracy={}'.format(*scores)
