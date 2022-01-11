from your_code import GradientDescent, load_data, L1Regularization, L2Regularization
from your_code import HingeLoss, SquaredLoss, ZeroOneLoss
from your_code import accuracy, confusion_matrix

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    # 1a 
    # load mnist-binary data with fraction 1.0
    train_features, test_features, train_targets, test_targets = load_data('mnist-binary', fraction=1.0)
    print(train_features.shape)

    # gradient descent with hinge loss and learning rate 1e-4
    grad_des = GradientDescent(loss='hinge',learning_rate = 1e-4)

    # fit on the training data and plot the loss and accuracy
    grad_des.fit(train_features, train_targets)

    plt.figure()
    plt.plot(range(len(grad_des.loss_fr)),grad_des.loss_fr)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss per iteration')
    plt.savefig('experiments/figures/loss_grad_des')
    
    plt.figure()
    plt.plot(range(len(grad_des.accuracy_fr)),grad_des.accuracy_fr)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per iteration')
    plt.savefig('experiments/figures/accuracy_grad_des')

    #1b max_iter 16000, for epoch calculate the gradient and plot the loss and accuracies
    train_features, test_features, train_targets, test_targets = load_data('mnist-binary', fraction=1.0)
    print(train_features.shape)
    grad_des = GradientDescent(loss='hinge', learning_rate = 1e-4)

    grad_des.epoch_fit(train_features, train_targets,batch_size = 64, max_iter = 16000)


    plt.figure()
    plt.plot(range(len(grad_des.loss_fr)),grad_des.loss_fr)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss per iteration')
    plt.savefig('experiments/figures/loss_grad_des_epoch')
    
    plt.figure()
    plt.plot(range(len(grad_des.accuracy_fr)),grad_des.accuracy_fr)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per iteration')
    plt.savefig('experiments/figures/accuracy_grad_des_epoch')






