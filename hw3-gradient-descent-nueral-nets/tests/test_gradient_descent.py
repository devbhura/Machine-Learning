import numpy as np
from your_code import load_data


def make_predictions(features, targets, loss, regularization):
    """
    Fit and predict on the training set using gradient descent and default
    parameter values. Note that in practice, the testing set should be used for
    predictions. This code is just to common-sense check that your gradient
    descent algorithm can classify the data it was trained on.
    """
    from your_code import GradientDescent

    np.random.seed(0)
    learner = GradientDescent(loss=loss, regularization=regularization,
                              learning_rate=0.01, reg_param=0.05)
    learner.fit(features, targets, batch_size=None, max_iter=1000)

    return learner.predict(features)


def test_gradient_descent_blobs():
    """
    Tests the ability of the gradient descent algorithm to classify a linearly
    separable dataset.
    """
    features, _, targets, _ = load_data('blobs')

    hinge = make_predictions(features, targets, 'hinge', None)
    assert np.all(hinge == targets)

    l1_hinge = make_predictions(features, targets, 'hinge', 'l1')
    assert np.all(l1_hinge == targets)

    l2_hinge = make_predictions(features, targets, 'hinge', 'l2')
    assert np.all(l2_hinge == targets)

    squared = make_predictions(features, targets, 'squared', None)
    assert np.all(squared == targets)

    l1_squared = make_predictions(features, targets, 'squared', 'l1')
    assert np.all(l1_squared == targets)

    l2_squared = make_predictions(features, targets, 'squared', 'l2')
    assert np.all(l2_squared == targets)


def test_gradient_descent_mnist_binary():
    """
    Tests the ability of the gradient descent classifier to classify a
    non-trivial problem with a reasonable accuracy.
    """
    from your_code import GradientDescent, accuracy

    train_features, test_features, train_targets, test_targets = \
        load_data('mnist-binary', fraction=0.8)

    accuracies = []
    for s in [0, 1, 13, 21, 234, 12093, 123901, 101, 101010, 1020]:
        np.random.seed(s)

        learner = GradientDescent(loss='squared', regularization=None,
                                learning_rate=0.01, reg_param=0.05)
        learner.fit(train_features, train_targets, batch_size=None, max_iter=1000)
        predictions = learner.predict(test_features)
    
        accuracies.append(accuracy(test_targets, predictions))
    
    accuracies = np.array(accuracies)
    assert accuracies.mean() > 0.95
