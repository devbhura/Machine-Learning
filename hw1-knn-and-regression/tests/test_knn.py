import numpy as np
from src import KNearestNeighbor, load_json_data
import os

def test_k_nearest_neighbor():
    from sklearn.metrics import accuracy_score as accuracy
    datasets = [
        os.path.join('data', x)
        for x in os.listdir('data')
        if os.path.splitext(x)[-1] == '.json'
    ]

    aggregators = ['mean', 'mode', 'median']
    distances = ['euclidean', 'manhattan']
    for data_path in datasets:
        # Load data and make sure its shape is correct
        features, targets = load_json_data(data_path)
        targets = targets[:, None]  # expand dims
        for d in distances:
            for a in aggregators:
                # make model and fit
                knn = KNearestNeighbor(1, distance_measure=d, aggregator=a)
                knn.fit(features, targets)

                # predict and calculate accuracy
                labels = knn.predict(features)
                acc = accuracy(targets, labels)

                # error if there's an issue
                msg = 'Failure with dataset: {}. Settings: dist={}, agg={}.'.format(data_path, d, a)
                assert (acc == 1.0), msg


def test_aggregators():
    _features = np.array([
        [-1, 1, 1, -1, 2],
        [-1, 1, 1, -1, 1],
        [-1, 2, 2, -1, 1],
        [-1, 1, 1, -1, 1],
        [-1, 1, 1, -1, 1]
    ])

    _predict = np.array([
        [-1, 1, 0, -1, 0],
        [-1, 1, 1, -1, 0],
        [-1, 0, 1, 0, 0],
        [-1, 1, 1, -1, 1],
        [-1, 1, 1, -1, 0]
    ])
    _targets = np.array([
        [1, 0, 1],
        [1, 1, 5],
        [3, 1, 1],
        [1, 1, 2],
        [5, 1, 1]
    ])
    aggregators = ['mean', 'mode', 'median']
    answers = [
        np.repeat(np.mean(_targets, axis=0, keepdims=True), _targets.shape[0], axis=0),
        np.ones_like(_targets),
        np.repeat(np.median(_targets, axis=0, keepdims=True), _targets.shape[0], axis=0)
    ]
    _est = []
    for a in aggregators:
        knn = KNearestNeighbor(5, aggregator=a)
        knn.fit(_features, _targets)
        y = knn.predict(_predict)
        _est.append(y)
    assert (np.allclose(_est, answers))
