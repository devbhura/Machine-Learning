import numpy as np


def test_l1_regularization_forward():
    """
    Test the forward pass of the L1Regularization class.
    """
    from your_code import L1Regularization

    X = np.array([[-1, 2, 1], [-3, 4, 1]])
    regularizer = L1Regularization(reg_param=0.5)

    _true = np.array([1.5, 3.5])
    _est = np.array([regularizer.forward(x) for x in X])

    assert np.allclose(_true, _est)


def test_l1_regularization_backward():
    """
    Test the backward pass of the L1Regularization class.
    """
    from your_code import L1Regularization

    X = np.array([[-1, 2, 1], [-3, 4, 1]])
    regularizer = L1Regularization(reg_param=0.5)

    _true = np.array([[-0.5, 0.5, 0], [-0.5, 0.5, 0]])
    _est = np.array([regularizer.backward(x) for x in X])

    assert np.allclose(_true, _est)


def test_l2_regularization_forward():
    """
    Test the forward pass of the L2Regularization class.
    """
    from your_code import L2Regularization

    X = np.array([[-1, 2, 1], [-3, 4, 1]])
    regularizer = L2Regularization(reg_param=0.5)

    _true = np.array([1.25, 6.25])
    _est = np.array([regularizer.forward(x) for x in X])

    assert np.allclose(_true, _est)


def test_l2_regularization_backward():
    """
    Test the backward pass of the L2Regularization class.
    """
    from your_code import L2Regularization

    X = np.array([[-1, 2, 1], [-3, 4, 1]])
    regularizer = L2Regularization(reg_param=0.5)

    _true = np.array([[-0.5, 1, 0], [-1.5, 2, 0]])
    _est = np.array([regularizer.backward(x) for x in X])

    assert np.allclose(_true, _est)
