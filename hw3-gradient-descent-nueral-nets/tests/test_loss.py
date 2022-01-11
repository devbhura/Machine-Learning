import numpy as np


def test_hinge_loss_forward():
    """
    Tests the forward pass of the hinge loss function
    """
    from your_code import HingeLoss
    X = np.array([[-1, 2, 1], [-3, 4, 1]])
    w = np.array([1, 2, 3])
    y = np.array([1, -1])

    loss = HingeLoss(regularization=None)

    _true = 4.5
    _est = loss.forward(X, w, y)

    assert np.allclose(_true, _est)


def test_hinge_loss_backward():
    """
    Tests the backward pass of the hinge loss function
    """
    from your_code import HingeLoss
    X = np.array([[-1, 2, 1], [-3, 4, 1]])
    w = np.array([1, 2, 3])
    y = np.array([1, -1])

    loss = HingeLoss(regularization=None)

    _true = np.array([-1.5, 2, 0.5])
    _est = loss.backward(X, w, y)

    assert np.allclose(_true, _est)


def test_squared_loss_forward():
    """
    Tests the forward pass of the squared loss function
    """
    from your_code import SquaredLoss
    X = np.array([[-1, 2, 1], [-3, 4, 1]])
    w = np.array([1, 2, 3])
    y = np.array([1, -1])

    loss = SquaredLoss(regularization=None)

    _true = 26.5
    _est = loss.forward(X, w, y)

    assert np.allclose(_true, _est)


def test_squared_loss_backward():
    """
    Tests the backward pass of the squared loss function
    """
    from your_code import SquaredLoss
    X = np.array([[-1, 2, 1], [-3, 4, 1]])
    w = np.array([1, 2, 3])
    y = np.array([1, -1])

    loss = SquaredLoss(regularization=None)

    _true = np.array([-16, 23, 7])
    _est = loss.backward(X, w, y)

    assert np.allclose(_true, _est)
