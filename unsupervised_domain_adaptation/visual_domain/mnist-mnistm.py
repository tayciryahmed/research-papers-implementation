import numpy as np
import cPickle as pkl
from sklearn.manifold import TSNE
from ATDA import ATDA
from utils import *
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../ATDA/MNIST_data', one_hot=True)
# Process MNIST
mnist_train = (mnist.train.images > 0).reshape(55000, 28, 28, 1).astype(np.float32)
mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)
mnist_test = (mnist.test.images > 0).reshape(10000, 28, 28, 1).astype(np.float32)
mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)


# Load MNIST-M
mnistm = pkl.load(open('../ATDA/mnistm_data.pkl'))
mnistm_train = mnistm['train']/255.
mnistm_test = mnistm['test']/255.
mnistm_valid = mnistm['valid']/255.

model = ATDA()

model.fit_ATDA(X_source_train=mnist_train, y_source_train=mnist.train.labels,
                       X_target_test=mnistm_test, y_target_test=mnist.test.labels,
                       X_target_train_valid=mnistm_train, y_target_train_valid=mnist.train.labels,
                      threshold=0.9, n_epoch=3000, k=30, lr=0.01, batch_size_F1F2=64, batch_size_Ft=128)

"""
model.fit_ATDA(X_source_train=mnist_train[:1000, :], y_source_train=mnist.train.labels[:1000],
                       X_target_test=mnistm_test[:1000, :], y_target_test=mnist.test.labels[:1000],
                       X_target_train_valid=mnistm_train[:1000, :], y_target_train_valid=mnist.train.labels[:1000],
                      threshold=0, n_epoch=3000, k=30, batch_size_F1F2=1, batch_size_Ft=1, lr=0.01)
"""
