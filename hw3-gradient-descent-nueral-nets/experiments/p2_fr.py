from your_code import GradientDescent, load_data, L1Regularization, L2Regularization
from your_code import HingeLoss, SquaredLoss, ZeroOneLoss
from your_code import accuracy, confusion_matrix

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    # 2a - load synthetic data
    train_features, test_features, train_targets, test_targets = load_data('synthetic', fraction=1.0)

    # define the bias
    bias = np.linspace(-5.5,0.5,100)
    N, d = np.shape(train_features)
    X = np.hstack((train_features,np.ones((N,1))))
    print(X.shape)
    loss_list = []
    for i in range(len(bias)):
        w = np.append(np.ones(d),bias[i])
        print(w)
        print(w.shape)
        y = train_targets
        predictions = (X @ w > 0.0).astype(int) * 2 - 1
        loss = np.sum((predictions != y).astype(float)) / len(X)
        loss_list.append(loss)

    plt.figure()
    plt.plot(bias,loss_list)
    plt.xlabel('Bias')
    plt.ylabel('Loss')
    plt.title('Loss per bias')
    plt.savefig('experiments/figures/q2')

    # 2b 
    index = [0,1,4,5]
    train_features_b = train_features[index]
    
    train_targets_b = train_targets[index]
    bias = np.linspace(-5.5,0.5,100)
    N, d = np.shape(train_features_b)
    X = np.hstack((train_features_b,np.ones((N,1))))
    
    loss_list = []
    for i in range(len(bias)):
        w = np.append(np.ones(d),bias[i])
        
        y = train_targets_b
        predictions = (X @ w > 0.0).astype(int) * 2 - 1
        loss = np.sum((predictions != y).astype(float)) / len(X)
        loss_list.append(loss)

    plt.figure()
    plt.plot(bias,loss_list)
    plt.xlabel('Bias')
    plt.ylabel('Loss')
    plt.title('Loss per bias')
    plt.savefig('experiments/figures/q2_b')


