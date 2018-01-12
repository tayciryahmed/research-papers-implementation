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

print mnist_train.shape
print mnist_test.shape
print mnistm_train.shape
print mnistm_test.shape
print mnist.train.labels.shape
print mnist.test.labels.shape

model = ATDA()
model.fit_ATDA(source_train=mnist_train, y_train=mnist.train.labels,
                       target_val=mnistm_test, y_val=mnist.test.labels,
                       target_data=mnistm_train,target_label=mnist.train.labels)
