import numpy as np
import cPickle as pkl
from ATDA import ATDA
from utils import *
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split


def document_preprocessor(doc):
    return str(doc[0])

src = pd.read_csv('kitchen.csv', sep='|')
tgt = pd.read_csv('cd.csv', sep='|')

X_source_train = src['reviewText'].values.reshape(-1,1)
y_source_train = src['overall'].values.reshape(-1,1)>3

#TFIDF
vectorizer = TfidfVectorizer(max_features=5000, preprocessor=document_preprocessor)
X_source_train = vectorizer.fit_transform(X_source_train).todense()



#split tgt (8000 samples: 6000 test samples)
X_target_train_valid_test = tgt['reviewText'].values.reshape(-1,1)
y_target_train_valid_test = tgt['overall'].values.reshape(-1,1)>3

#TFIDF
vectorizer = TfidfVectorizer(max_features=5000, preprocessor=document_preprocessor)
X_target_train_valid_test = vectorizer.fit_transform(X_target_train_valid_test).todense()

X_target_train_valid, X_target_test, y_target_train_valid,y_target_test  = \
            train_test_split(X_target_train_valid_test, y_target_train_valid_test, test_size=0.75, random_state=42)

X_source_train = np.asarray(X_source_train).reshape(len(X_source_train), 25,25,8)
X_target_test = np.asarray(X_target_test).reshape(len(X_target_test), 25,25,8)
X_target_train_valid = np.asarray(X_target_train_valid).reshape(len(X_target_train_valid), 25,25,8)

print X_source_train.shape, y_source_train.shape, \
    X_target_test.shape, y_target_test.shape, \
    X_target_train_valid.shape, y_target_train_valid.shape

model = ATDA()

model.fit_ATDA(X_source_train=X_source_train, y_source_train=y_source_train,
                       X_target_test=X_target_test, y_target_test=y_target_test,
                       X_target_train_valid=X_target_train_valid, y_target_train_valid=y_target_train_valid,
                      threshold=0.9, n_epoch=3000, k=30, lr=0.01, batch_size_F1F2=64, batch_size_Ft=128)
