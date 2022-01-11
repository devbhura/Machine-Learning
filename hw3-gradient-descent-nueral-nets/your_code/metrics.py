import numpy as np


def accuracy(ground_truth, predictions):
    """
    Reports the classification accuracy.

    Arguments:
        ground_truth - (np.array) A 1D numpy array of length N. The true class
            labels.
        predictions - (np.array) A 1D numpy array of length N. The class labels
            predicted by the model.
    Returns:
        accuracy - (float) The accuracy of the predictions.
    """
    return np.mean(ground_truth == predictions)


def confusion_matrix(ground_truth, predictions):
    """
    Reports the classification accuracy.

    Arguments:
        ground_truth - (np.array) A 1D numpy array of length N. The true class
            labels.
        predictions - (np.array) A 1D numpy array of length N. The class labels
            predicted by the model.
    Returns:
        confusion_matrix - (np.array) The confusion matrix. A CxC numpy array,
            where C is the number of unique classes. Index i, j is the number
            of times an example belonging to class i was predicted to belong
            to class j.
    """
    classes = np.unique(ground_truth)
    confusion = np.zeros((len(classes), len(classes)))
    for i, prediction in enumerate(predictions):
        confusion[ground_truth[i], prediction] += 1
    return confusion
