import numpy as np 
from matplotlib import pyplot as plt
from generate_regression_data import generate_regression_data
from polynomial_regression import PolynomialRegression
from k_nearest_neighbor import KNearestNeighbor
from metrics import mean_squared_error
from load_json_data import load_json_data
import os

from mpl_toolkits.mplot3d import Axes3D

def clear_spiral():
    
    features, target = load_json_data("/home/devbhura/fall_2021/ML/HW1/hw1-knn-and-regression-devbhura/data/clean-spiral.json")
    # expand dims
    target = target[:, None]
    print(target)
    

    k = [1]
    error_train = []
    error_test = []
    for i in k:

        knn = KNearestNeighbor(i, distance_measure='euclidean', aggregator='mean')
        knn.fit(features, target)
        label = knn.predict(features)
        print(label)
        # mse_train = mean_squared_error(y_train_B,y_hat)
        # error_train.extend([mse_train])
        
        plt.figure(figsize=(6, 4))
        plt.scatter(features[:, 0], features[:, 1], c=target)
        
        

        # xs, ys = zip(*sorted(zip(features, label)))
        # plt.plot(features[:, 0], features[:, 1], c=label)

        # print(f"x_train={x_train_A}")
        # print(f"y_train={y_train_A}")
        # print(f"y_hat={y_hat}")

        
        plt.legend(["Spiral"])
        plt.title(f"Curves for kNN = {i}")
        plt.savefig(f"/home/devbhura/fall_2021/ML/HW1/hw1-knn-and-regression-devbhura/src/figures/Spiral{i}.png")

        # mse_test = mean_squared_error(y_test_B,y_hat)
        # error_test.extend([mse_test])
       
    # indextrain = error_train.index(min(error_train))
    # indextest = error_test.index(min(error_test))

    # plt.figure()
    # plt.scatter(x_train_B,y_train_B)
    # i_train = degree[indextrain]
    # i_test = degree[indextest]

    # p = PolynomialRegression(i_train)
    # p.fit(x_train_B, y_train_B)
    # y_hat = p.predict(x_train_B)

    # xs, ys = zip(*sorted(zip(x_train_B, y_hat)))
    # plt.plot(xs,ys)

    # p = PolynomialRegression(i_test)
    # p.fit(x_train_B, y_train_B)
    # y_hat = p.predict(x_test_B)

    # xs, ys = zip(*sorted(zip(x_test_B, y_hat)))
    # plt.plot(xs,ys)

    # plt.legend([f"Lowest Training Error deg={i_train} ",f"Lowest Test Error deg={i_test} "])
    # plt.title(f"Polyfit with lowest error")
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.savefig(f"/home/devbhura/fall_2021/ML/HW1/hw1-knn-and-regression-devbhura/src/figures/PolyfitMinB.png")

    # plt.figure()
    # plt.plot(degree,np.log(error_train))
    # plt.plot(degree,np.log(error_test))
    # plt.title("Figure 1. Testing and Training Error Graphs with degree dependency")
    # plt.xlabel("Degree")
    # plt.ylabel("Error")
    # plt.legend(["Training Error","Testing Error"])
    # plt.savefig('/home/devbhura/fall_2021/ML/HW1/hw1-knn-and-regression-devbhura/src/figures/ErrorvsDegreeB.png')

    return None


clear_spiral()