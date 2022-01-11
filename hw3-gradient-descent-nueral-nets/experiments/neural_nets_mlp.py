from sklearn.neural_network import MLPClassifier

from your_code import GradientDescent, load_data, load_mnist_data
import warnings
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
import numpy as np
import matplotlib.pyplot as plt

"""Load mnist data
Calculates the best hidden layer paramaters (hidden_layer_nodes, activation fnc, and alpha values) by running for loops and calculating accuracies
"""
train_features, test_features, train_targets, test_targets = load_mnist_data(threshold=10, fraction=0.9)

print("Dataset loaded")

hidden_layer_nodes = [1,4,16,64,256]

acc_layer_mean = []
acc_layer_std = []

for i in hidden_layer_nodes:
    print("Hidden Layer:",i)
    acc_list = []
    for j in range(10):
        print("Loop",j)
        clf = MLPClassifier(hidden_layer_sizes=(i,))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
            clf.fit(train_features, train_targets)

        
        acc = clf.score(test_features,test_targets)
        acc_list.append(acc)
    
    acc_layer_mean.append(np.mean(np.array(acc_list)))
    acc_layer_std.append(np.std(np.array(acc_list)))

    
print(acc_layer_mean)
print(acc_layer_std)

print("Done")

# choose hidden layer 16 as it gives perfect accuracy on the training dataset for the least number of nodes in the hidden layer

# hidden_layer = 256

# act_fnc = ['logistic','identity','tanh','relu']

# acc_layer_mean = []
# acc_layer_std = []

# for i in act_fnc:
#     print("Activation Fnc:",i)
#     acc_list = []
#     for j in range(10):
#         print("Loop",j)
#         clf = MLPClassifier(hidden_layer_sizes=(hidden_layer,),activation= i)
#         with warnings.catch_warnings():
#             warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
#             clf.fit(train_features, train_targets)

        
#         acc = clf.score(test_features,test_targets)
#         acc_list.append(acc)
    
#     acc_layer_mean.append(np.mean(np.array(acc_list)))
#     acc_layer_std.append(np.std(np.array(acc_list)))

    
# print(acc_layer_mean)
# print(acc_layer_std)

# print("Done")


######### 7c hidden_layer_size = 256, act_fnc = 'relu'

# hidden_layer = 256

# act_fnc = 'relu'


# clf_1 = MLPClassifier(hidden_layer_sizes=(hidden_layer,),activation= act_fnc,alpha=1)
# clf_2 = MLPClassifier(hidden_layer_sizes=(hidden_layer,),activation= act_fnc,alpha=0.1)
# clf_3 = MLPClassifier(hidden_layer_sizes=(hidden_layer,),activation= act_fnc,alpha=0.01)
# clf_4 = MLPClassifier(hidden_layer_sizes=(hidden_layer,),activation= act_fnc,alpha=0.001)
# clf_5 = MLPClassifier(hidden_layer_sizes=(hidden_layer,),activation= act_fnc,alpha=0.0001)

# l2_reg = [1, 0.1,0.01,0.001,0.0001]
# acc_layer_mean = []
# acc_layer_std = []

# acc_1_list = []
# acc_2_list = []
# acc_3_list = []
# acc_4_list = []
# acc_5_list = []

# for j in range(10):
#     print("Loop",j)
    
#     with warnings.catch_warnings():
#         warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
#         clf_1.fit(train_features, train_targets)
#         clf_2.fit(train_features, train_targets)
#         clf_3.fit(train_features, train_targets)
#         clf_4.fit(train_features, train_targets)
#         clf_5.fit(train_features, train_targets)

    
#     acc_1 = clf_1.score(test_features,test_targets)
#     acc_2 = clf_2.score(test_features,test_targets)
#     acc_3 = clf_3.score(test_features,test_targets)
#     acc_4 = clf_4.score(test_features,test_targets)
#     acc_5 = clf_5.score(test_features,test_targets)
    
#     acc_1_list.append(acc_1)
#     acc_2_list.append(acc_2)
#     acc_3_list.append(acc_3)
#     acc_4_list.append(acc_4)
#     acc_5_list.append(acc_5)

# means = [np.mean(np.array(acc_1_list)),np.mean(np.array(acc_2_list)),np.mean(np.array(acc_3_list)),np.mean(np.array(acc_4_list)),np.mean(np.array(acc_5_list))]
# stddev = [np.std(np.array(acc_1_list)),np.std(np.array(acc_2_list)),np.std(np.array(acc_3_list)),np.std(np.array(acc_4_list)),np.std(np.array(acc_5_list))]


    
# print(means)
# print(stddev)

# print("Done")


##### Problem 8 Visualize the data

clf = MLPClassifier(hidden_layer_sizes=(256,),activation= 'relu',alpha=1)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    clf.fit(train_features, train_targets)
fig, axes = plt.subplots(6, 6)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = clf.coefs_[0].min(), clf.coefs_[0].max()
for coef, ax in zip(clf.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=0.5 * vmin, vmax=0.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.savefig('experiments/figures/p8')