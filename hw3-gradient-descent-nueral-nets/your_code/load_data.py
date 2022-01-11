import json
import numpy as np
import os
import struct
from array import array as pyarray


def load_data(dataset, fraction=1.0, base_folder='data'):
    """
    Loads a dataset and performs a random stratified split into training and
    test partitions.

    Arguments:
        dataset - (string) The name of the dataset to load. One of the
            following:
              'blobs': A linearly separable binary classification problem.
              'mnist-binary': A subset of the MNIST dataset containing only
                  0s and 1s.
              'mnist-multiclass': A subset of the MNIST dataset containing the
                  numbers 0 through 4, inclusive.
              'synthetic': A small custom dataset for exploring properties of
                  gradient descent algorithms.
        fraction - (float) Value between 0.0 and 1.0 representing the fraction
            of data to include in the training set. The remaining data is
            included in the test set. Unused if dataset == 'synthetic'.
        base_folder - (string) absolute path to your 'data' directory. If
            defaults to 'data'.
    Returns:
        train_features - (np.array) An Nxd array of features, where N is the
            number of examples and d is the number of features.
        test_features - (np.array) An Nxd array of features, where M is the
            number of examples and d is the number of features.
        train_targets - (np.array) A 1D array of targets of size N.
        test_targets - (np.array) A 1D array of targets of size M.
    """
    if dataset == 'blobs':
        path = os.path.join(base_folder, 'blobs.json')
        train_features, test_features, train_targets, test_targets = \
            load_json_data(path)
    elif dataset == 'mnist-binary':
        train_features, test_features, train_targets, test_targets = \
            load_mnist_data(2, fraction=fraction, mnist_folder=base_folder)
        train_targets = train_targets * 2 - 1
        test_targets = test_targets * 2 - 1
    elif dataset == 'mnist-multiclass':
        train_features, test_features, train_targets, test_targets = \
            load_mnist_data(5, fraction=fraction, examples_per_class=100,
                            mnist_folder=base_folder)
    elif dataset == 'synthetic':
        path = os.path.join(base_folder,  'synthetic.json')
        train_features, test_features, train_targets, test_targets = \
            load_json_data(path)
    else:
        raise ValueError('Dataset {} not found!'.format(dataset))

    # Normalize the data using feature-independent whitening. Note that the
    # statistics are computed with respect to the training set and applied to
    # both the training and testing sets.
    if dataset != 'synthetic':
        mean = train_features.mean(axis=0, keepdims=True)
        std = train_features.std(axis=0, keepdims=True) + 1e-5
        train_features = (train_features - mean) / std
        if fraction < 1.0:
            test_features = (test_features - mean) / std

    return train_features, test_features, train_targets, test_targets


def load_json_data(path, fraction=None, examples_per_class=None):
    """
    Loads a dataset stored as a JSON file. This will not split your dataset
    into training and testing sets, rather it returns all features and targets
    in `train_features` and `train_targets` and leaves `test_features` and
    `test_targets` as empty numpy arrays. This is done to match the API
    of the other data loaders.

    Args:
        path - (string) Path to json file containing the data
        fraction - (float) Ignored.
        examples_per_class - (int) - Ignored.

    Returns:
        train_features - (np.array) An Nxd array of features, where N is the
            number of examples and d is the number of features.
        test_features - (np.array) An empty 2D numpy array.
        train_targets - (np.array) A 1D array of targets of size N.
        test_targets - (np.array) An empty 1D array.
    """
    with open(path, 'rb') as file:
        data = json.load(file)
    features = np.array(data[0]).astype(float)
    targets = np.array(data[1]).astype(int)

    return features, np.array([[]]), targets, np.array([])


def load_mnist_data(threshold=10, fraction=1.0, examples_per_class=500, mnist_folder='data'):
    """
    Loads a subset of the MNIST dataset.

    Arguments:
        threshold - (int) One greater than the maximum digit in the selected
            subset. For example to get digits [0, 1, 2] this arg should be 3, or
            to get the digits [0, 1, 2, 3, 4, 5, 6] this arg should be 7.
        fraction - (float) Value between 0.0 and 1.0 representing the fraction
            of data to include in the training set. The remaining data is
            included in the test set. Unused if dataset == 'synthetic'.
        examples_per_class - (int) Number of examples to retrieve in each
            class.
        mnist_folder - (string) Path to folder contain MNIST binary files.

    Returns:
        train_features - (np.array) An Nxd array of features, where N is the
            number of examples and d is the number of features.
        test_features - (np.array) An Nxd array of features, where M is the
            number of examples and d is the number of features.
        train_targets - (np.array) A 1D array of targets of size N.
        test_targets - (np.array) A 1D array of targets of size M.
    """
    assert 0.0 <= fraction <= 1.0, 'Whoopsies! Incorrect value for fraction :P'

    train_examples = int(examples_per_class * fraction)
    if train_examples == 0:
        train_features, train_targets = np.array([[]]), np.array([])
    else:
        train_features, train_targets = _load_mnist(
            dataset='training', digits=range(threshold), path=mnist_folder)
        train_features, train_targets = stratified_subset(
            train_features, train_targets, train_examples)
        train_features = train_features.reshape((len(train_features), -1))

    test_examples = examples_per_class - train_examples
    if test_examples == 0:
        test_features, test_targets = np.array([[]]), np.array([])
    else:
        test_features, test_targets = _load_mnist(
            dataset='testing', digits=range(threshold), path=mnist_folder)
        test_features, test_targets = stratified_subset(
            test_features, test_targets, test_examples)
        test_features = test_features.reshape((len(test_features), -1))

    return train_features, test_features, train_targets, test_targets


def _load_mnist(path, dataset="training", digits=None, asbytes=False,
                selection=None, return_labels=True, return_indices=False):
    """
    Loads MNIST files into a 3D numpy array. Does not automatically download
    the dataset. You must download the dataset manually. The data can be
    downloaded from http://yann.lecun.com/exdb/mnist/.

    Examples:
        1) Assuming that you have downloaded the MNIST database in a directory
        called 'data', this will load all images and labels from the training
        set:

            images, labels = _load_mnist('training')

        2) And this will load 100 sevens from the test partition:

            sevens = _load_mnist('testing', digits=[7], selection=slice(0, 100),
                                return_labels=False)

    Arguments:
        path - (str) Path to your MNIST datafiles.
        dataset - (str) Either "training" or "testing". The data partition to
            load.
        digits - (list or None) A list of integers specifying the digits to
            load. If None, the entire database is loaded.
        asbytes - (bool) If True, returns data as ``numpy.uint8`` in [0, 255]
            as opposed to ``numpy.float64`` in [0.0, 1.0].
        selection - (slice) Using a `slice` object, specify what subset of the
            dataset to load. An example is ``slice(0, 20, 2)``, which would
            load every other digit until--but not including--the twentieth.
        return_labels - (bool) Specify whether or not labels should be
            returned. This is also a speed performance if digits are not
            specified, since then the labels file does not need to be read at
            all.
        return_indicies - (bool) Specify whether or not to return the MNIST
            indices that were fetched. This is valuable only if digits is
            specified, because in that case it can be valuable to know how far
            in the database it reached.
    Returns:
        images - (np.array) Image data of shape ``(N, rows, cols)``, where
            ``N`` is the number of images. If neither labels nor indices are
            returned, then this is returned directly, and not inside a 1-sized
            tuple.
        labels - (np.array) Array of size ``N`` describing the labels.
            Returned only if ``return_labels`` is `True`, which is default.
        indices - (np.array) The indices in the database that were returned.
    """

    # The files are assumed to have these names and should be found in 'path'
    files = {
        'training': ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte'),
        'testing': ('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte'),
    }

    try:
        images_fname = os.path.join(path, files[dataset][0])
        labels_fname = os.path.join(path, files[dataset][1])
    except KeyError:
        raise ValueError("Data set must be 'testing' or 'training'")

    # We can skip the labels file only if digits aren't specified and labels
    # aren't asked for
    if return_labels or digits is not None:
        flbl = open(labels_fname, 'rb')
        magic_nr, size = struct.unpack(">II", flbl.read(8))
        labels_raw = pyarray("b", flbl.read())
        flbl.close()

    fimg = open(images_fname, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    images_raw = pyarray("B", fimg.read())
    fimg.close()

    if digits:
        indices = [k for k in range(size) if labels_raw[k] in digits]
    else:
        indices = range(size)

    if selection:
        indices = indices[selection]

    images = np.zeros((len(indices), rows, cols), dtype=np.uint8)

    if return_labels:
        labels = np.zeros((len(indices)), dtype=np.int8)
    for i in range(len(indices)):
        images[i] = np.array(images_raw[indices[i] * rows * cols:(indices[i] + 1) * rows * cols]).reshape((rows, cols))
        if return_labels:
            labels[i] = labels_raw[indices[i]]

    if not asbytes:
        images = images.astype(float)/255.0

    ret = (images,)
    if return_labels:
        ret += (labels,)
    if return_indices:
        ret += (indices,)

    if len(ret) == 1:
        return ret[0]  # Don't return a tuple of one

    return ret


def stratified_subset(features, targets, examples_per_class):
    """
    Evenly sample the dataset across unique classes. Requires each unique class
    to have at least examples_per_class examples.

    Arguments:
        features - (np.array) An Nxd array of features, where N is the
            number of examples and d is the number of features.
        targets - (np.array) A 1D array of targets of size N.
        examples_per_class - (int) The number of examples to take in each
            unique class.
    Returns:
        train_features - (np.array) An Nxd array of features, where N is the
            number of examples and d is the number of features.
        test_features - (np.array) An Nxd array of features, where M is the
            number of examples and d is the number of features.
        train_targets - (np.array) A 1D array of targets of size N.
        test_targets - (np.array) A 1D array of targets of size M.
    """
    idxs = np.array([False] * len(features))
    for target in np.unique(targets):
        idxs[np.where(targets == target)[0][:examples_per_class]] = True
    return features[idxs], targets[idxs]
