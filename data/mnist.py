import os
import gzip
import tarfile
import zipfile
import math

import numpy as onp
import pickle
import urllib.request


def download_dataset(url, path):
    print('Downloading data from %s' % url)
    urllib.request.urlretrieve(url, path)


def to_one_hot(x, depth):
    """
    Get one-hot representation of a 1-D numpy array of integers.
    :param x: 1-D Numpy array of type int.
    :param depth: A int.
    :return: 2-D Numpy array of type int.
    """
    ret = onp.zeros((x.shape[0], depth))
    ret[onp.arange(x.shape[0]), x] = 1
    return ret


def load_mnist_realval(path, one_hot=True, dequantify=False):
    """
    Loads the real valued MNIST dataset.
    :param path: Path to the dataset file.
    :param one_hot: Whether to use one-hot representation for the labels.
    :param dequantify:  Whether to add uniform noise to dequantify the data
        following (Uria, 2013).
    :return: The MNIST dataset.
    """
    if not os.path.isfile(path):
        data_dir = os.path.dirname(path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(data_dir)
        download_dataset('http://ml.cs.tsinghua.edu.cn/~ziyu/static'
                         '/mnist.pkl.gz', path)

    with gzip.open(path, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    x_train, t_train = train_set[0], train_set[1]
    x_valid, t_valid = valid_set[0], valid_set[1]
    x_test, t_test = test_set[0], test_set[1]
    if dequantify:
        x_train += onp.random.uniform(0, 1. / 256,
                                     size=x_train.shape).astype('float32')
        x_valid += onp.random.uniform(0, 1. / 256,
                                     size=x_valid.shape).astype('float32')
        x_test += onp.random.uniform(0, 1. / 256,
                                    size=x_test.shape).astype('float32')
    n_y = t_train.max() + 1
    t_transform = (lambda x: to_one_hot(x, n_y)) if one_hot else (lambda x: x)
    return (x_train, t_transform(t_train)), (x_valid, t_transform(t_valid)), \
        (x_test, t_transform(t_test))
