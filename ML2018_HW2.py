import numpy as np
import sys, io
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
from sklearn.svm import NuSVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata("MNIST original")


X, Y = mnist.data, mnist.target

X, Y = shuffle(X, Y)

pca = PCA(svd_solver="arpack", n_components=69)

X = pca.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=1/10,random_state=None)

X_train = X_train.astype("float") / 255 * 2 - 1
X_test = X_test.astype("float") / 255 * 2 - 1

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

clf = NuSVC()
clf.fit(X_train, Y_train)
TEST=clf.predict(X_test)
TRAIN=clf.predict(X_train)
count=0
for i in range(Y_train.shape[0]):
    if TRAIN[i]==Y_train[i]:
        count=count+1

print(count/Y_train.shape[0])


count=0
for i in range(Y_test.shape[0]):
    if TEST[i]==Y_test[i]:
        count=count+1

print(count/Y_test.shape[0])
