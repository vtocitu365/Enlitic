import os
import gzip
import numpy as np
import cPickle as cp
import matplotlib.pyplot as plt

from pylab import imshow, show, cm
import time
from math import *

#---------------------------------------------------------------------------
# Code used to open and download the MNIST and CIFAR100 data
#---------------------------------------------------------------------------

# Loads a file
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cp.load(fo)
    return dict

# Opens a .pkl file and uploads the MNIST pictures. It returns the training,
# validation, and test data
def load_mnistData():
    with gzip.open('./mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = cp.load(f)
        [X_train, Y_train] = [train_set[0], train_set[1]]
        [X_valid, Y_valid] = [valid_set[0], valid_set[1]]
        [X_test, Y_test] = [test_set[0], test_set[1]]
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

# Opens batches of  file and uploads the CIFAR pictures. It returns the
# training, validation, and test data
def load_cifardata():
    dirname = os.path.join("", 'cifar-10-batches-py')

    # Opens and uploads each batch of the CIFAR100 training data
    for i in range(1, 6):
        fpath = os.path.join(dirname, 'data_batch_' + str(i))
        dict = unpickle(fpath)
        data = dict["data"]
        labels = dict["labels"]
        if i == 1:
            X_trainingSet = data
            Y_trainingSet = labels
        else:
            X_trainingSet = np.concatenate([X_trainingSet/255, data], axis=0)
            Y_trainingSet = np.concatenate([Y_trainingSet, labels], axis=0)

    fpath = os.path.join(dirname, 'test_batch')
    dict = unpickle(fpath)
    [X_valid, Y_valid] = [X_trainingSet[10000:20000], Y_trainingSet[10000:20000]]
    [X_train, Y_train] = [X_trainingSet[:10000], Y_trainingSet[:10000]]
    X_test = dict["data"]/255
    Y_test = dict["labels"]
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

#---------------------------------------------------------------------------
# Code used to visualize the datasets similar to the way it was done in
# Section 3 of the worksample document
#---------------------------------------------------------------------------

def visualize(X, n=20, data_img_channels=1):
    N = n**2
    D = data_img_channels
    dimensions = int(np.sqrt(len(X[0]) / D)) # This assumes that the pictures are square.
    montage = X[0:N].reshape(N, D, dimensions, dimensions).reshape(n, n, D, dimensions, dimensions).\
        transpose(0, 1, 3, 4, 2)
    img = montage.swapaxes(1, 2).reshape(dimensions*n, dimensions*n, D)
    if D == 1:
        img = img.reshape(dimensions*n, dimensions*n)
    fig = imshow(img,cmap = cm.gray)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    return plt

#---------------------------------------------------------------------------
# Code computes the log-probability of both datasets
#---------------------------------------------------------------------------

# Calculates the Gaussian Kernel, returning the log-probability
def kernel(x_a,x_b,sigma):
    len_t = len(x_a)
    len_v = len(x_b)
    d = len(x_a[0].astype(np.float64))
    mu = x_a.astype(np.float64)
    ln_k = np.log(len_t)
    sig = 2*(sigma**2)
    t2 = -(0.5*d)*np.log((np.pi*sig))
    p_x = 0
    for i in range(len_t):
        x = x_b[i].astype(np.float64)
        t1 = np.sum((-(x-mu)**2)/sig, axis=1)
        a = np.max(t1)
        sum_ele = np.sum(np.exp(t1 - a))
        log_p = a + np.log(sum_ele)
        p_x += log_p
    y = t2 - ln_k +p_x / len_v
    return y

sigma_range = [0.050, 0.080, 0.10, 0.20, 0.50, 1.0, 1.50, 2.0]

# Get the training, validation, and test datasets from the CIFAR100 dataset
[X_train, Y_train, X_valid, Y_valid, X_test, Y_test]=load_cifardata()
# Visualize the CIFAR100 dataset
img = visualize(X_train, data_img_channels=3)
# Grid-search for the optimal value of sigma, returning the log-probability
# of each sigma
L = [kernel(X_train[0:10000],X_valid[0:10000],sigma) for sigma in sigma_range]
print L
# Find the log-probability where the optimal sigma = 0.2
Lmax = kernel(X_train[0:10000],X_test[0:10000],0.2)
print Lmax

# Get the training, validation, and test datasets from the MNIST dataset
[X_train, Y_train, X_valid, Y_valid, X_test, Y_test] = load_mnistData()
# Visualize the MNIST dataset
plt.figure()
img = visualize(X_train, n=20, data_img_channels=1)
# Grid-search for the optimal value of sigma, returning the log-probability
# of each sigma
L = [kernel(X_train[0:10000],X_valid[0:10000],sigma) for sigma in sigma_range]
print L
# Find the log-probability where the optimal sigma = 0.2
Lmax = kernel(X_train[0:10000],X_test[0:10000],0.2)
print Lmax

#show()