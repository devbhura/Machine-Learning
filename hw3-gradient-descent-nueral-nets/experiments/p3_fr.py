from your_code import GradientDescent, load_data, L1Regularization, L2Regularization
from your_code import HingeLoss, SquaredLoss, ZeroOneLoss
from your_code import accuracy, confusion_matrix

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    train_features, test_features, train_targets, test_targets = load_data('mnist-binary', fraction=1.0)

    

    lam = [1e-3,1e-2,1e-1,1,10,100]
    w_n = []
    for reg in lam:
        grad_des = GradientDescent(loss='squared', learning_rate = 1e-5,reg_param=reg,regularization='l1')
        grad_des.fit(train_features, train_targets,max_iter= 2000)
        w = grad_des.model
        num = (abs(w)>0.001).sum()
        w_n.append(num)

    plt.figure()
    plt.plot(lam,w_n,label='l1')
    plt.xlabel('Lambda')
    plt.ylabel('Number of Non zero model values')
    plt.title('Lambda vs non-zero model val')
    # plt.savefig('experiments/figures/q3_l1')

    w_n_l2 = []
    for reg in lam:
        grad_des = GradientDescent(loss='squared', learning_rate = 1e-5,reg_param=reg,regularization='l2')
        grad_des.fit(train_features, train_targets,max_iter= 2000)
        w = grad_des.model
        num = (abs(w)>0.001).sum()
        w_n_l2.append(num)

    # plt.figure()
    plt.plot(lam,w_n_l2,label='l2')
    plt.legend()
    # plt.xlabel('Lambda')
    # plt.ylabel('Number of Non zero model values')
    # plt.title('Lambda vs non-zero model val - L2')
    plt.savefig('experiments/figures/q3')

    
    