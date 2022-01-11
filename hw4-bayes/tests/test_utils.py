import numpy as np

def test_softmax():
    from src.utils import softmax

    pairs = [
        ([0.5, 0.5], [0.5, 0.5]),
        ([-1e30, -1e30], [0.5, 0.5]),
        ([-12, -13], [0.73105858, 0.26894142]),
        ([-873.1424779, -847.35339601], [6.30876052e-12, 1.0]),
    ]

    for inp, out in pairs:
        est_ = softmax(np.array(inp), axis=1)
        # The softmax output should sum to 1
        assert np.isclose(np.sum(est_), 1)
        # The softmax should match reference
        assert np.allclose(est_, np.array(out))
