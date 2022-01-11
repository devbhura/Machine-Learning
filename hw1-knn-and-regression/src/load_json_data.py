import json
import numpy as np
import os


def load_json_data(json_path):
    """
    Loads data from JSON files kept in data/. Implemented this for you, you are
    welcome.

    Args:
        json_path (str): path to json file containing the data (e.g. 'data/blobs.json')
    Returns:
        features (np.ndarray): numpy array containing the x values
        targets (np.ndarray): numpy array containing the y values in the range -1, 1.
    """

    with open(json_path, 'rb') as f:
        data = json.load(f)
    features = np.array(data[0]).astype(float)
    targets = 2 * (np.array(data[1]).astype(float) - 1) - 1

    return features, targets


if __name__ == "__main__":
    """
    Running this from the command line in this directory will tell you the shapes of 
    each dataset, and also visualize the datasets for you.

        $ python load_json_data.py 
        (110, 2) (110,) ../data/parallel_lines.json
        (127, 2) (127,) ../data/blobs.json
        (68, 2) (68,) ../data/crossing.json
        (131, 2) (131,) ../data/circles.json

    """
    try:
        import matplotlib.pyplot as plt
    except:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

    data_files = [
        os.path.join('..', 'data', x)
        for x in os.listdir(os.path.join('..', 'data'))
        if os.path.splitext(x)[1] == '.json'
    ]

    for data_file in data_files:
        features, targets = load_json_data(data_file)
        plt.figure(figsize=(6, 4))
        plt.scatter(features[:, 0], features[:, 1], c=targets)
        plt.title(data_file)
        plt.savefig(os.path.join('..', 'data', '{}.png'.format(data_file)))
        print(features.shape, targets.shape, data_file)
