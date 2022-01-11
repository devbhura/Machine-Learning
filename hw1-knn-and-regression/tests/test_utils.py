import numpy as np
import random, string
import csv

def make_fake_data():
    y_true = np.random.random(100)
    y_pred = np.random.random(100)
    return y_pred, y_true

def write_random_csv_file(n_features, n_samples):
    from sklearn.datasets import make_classification
    features, targets = make_classification(
        n_samples=n_samples, 
        n_features=n_features, 
        n_classes=2
    )
    attribute_names = [
        random_string(10) for i in range(n_features)
    ]
    
    with open('tests/test.csv', 'w', newline='') as csvfile:
        writer = csv.writer(
            csvfile, 
            delimiter=',', 
            quotechar='|', 
            quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow(attribute_names + ['class'])
        for i in range(features.shape[0]):
            row = features[i].tolist() + [targets[i].tolist()]
            writer.writerow(row)

    return features, targets, attribute_names

def random_string(N):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))
