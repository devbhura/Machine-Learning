import numpy as np 
from matplotlib import pyplot as plt
from generate_regression_data import generate_regression_data
from polynomial_regression import PolynomialRegression
from k_nearest_neighbor import KNearestNeighbor
from metrics import mean_squared_error
import os

my_path = os.path.abspath(__file__)


def poly_reg():

    N = 100
    degree = 4
    features, target = generate_regression_data(degree, N, amount_of_noise=0.1)
    
    splitA = np.random.choice(N,N, replace=False)

    x_train_A, y_train_A  = features[splitA[:10]], target[splitA[:10]]
    x_test_A, y_test_A = features[splitA[10:]], target[splitA[10:]]


    splitB = np.random.choice(N,N, replace=False)
    x_train_B, y_train_B  = features[splitB[:50]], target[splitB[:50]]
    x_test_B, y_test_B = features[splitB[50:]], target[splitB[50:]]


    # For SplitA
    degree = np.arange(10)
    error_train = []
    error_test = []
    for i in degree:

        p = PolynomialRegression(i)
        p.fit(x_train_A, y_train_A)
        y_hat = p.predict(x_train_A)
        mse_train = mean_squared_error(y_train_A,y_hat)
        error_train.extend([mse_train])
        
        plt.figure()
        plt.scatter(x_train_A,y_train_A)
        xs, ys = zip(*sorted(zip(x_train_A, y_hat)))
        plt.plot(xs,ys)

        # print(f"x_train={x_train_A}")
        # print(f"y_train={y_train_A}")
        # print(f"y_hat={y_hat}")

        p.fit(x_train_A, y_train_A)
        
        y_hat = p.predict(x_test_A)
        xs, ys = zip(*sorted(zip(x_test_A, y_hat)))
        plt.plot(xs,ys)
        plt.legend(["Polyfit train","Polyfit test"])
        plt.title(f"Polynomial Curves for degree = {i}")
        plt.savefig(f"/home/devbhura/fall_2021/ML/HW1/hw1-knn-and-regression-devbhura/src/figures/PolyfitDeg{i}.png")

        mse_test = mean_squared_error(y_test_A,y_hat)
        error_test.extend([mse_test])
       
    indextrain = error_train.index(min(error_train))
    indextest = error_test.index(min(error_test))

    plt.figure()
    plt.scatter(x_train_A,y_train_A)
    i_train = degree[indextrain]
    i_test = degree[indextest]

    p = PolynomialRegression(i_train)
    p.fit(x_train_A, y_train_A)
    y_hat = p.predict(x_train_A)

    xs, ys = zip(*sorted(zip(x_train_A, y_hat)))
    plt.plot(xs,ys)

    p = PolynomialRegression(i_test)
    p.fit(x_train_A, y_train_A)
    y_hat = p.predict(x_test_A)

    xs, ys = zip(*sorted(zip(x_test_A, y_hat)))
    plt.plot(xs,ys)

    plt.legend([f"Lowest Training Error deg={i_train} ",f"Lowest Test Error deg={i_test} "])
    plt.title(f"Polyfit with lowest error")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f"/home/devbhura/fall_2021/ML/HW1/hw1-knn-and-regression-devbhura/src/figures/PolyfitMinA.png")

    plt.figure()
    plt.plot(degree,np.log(error_train))
    plt.plot(degree,np.log(error_test))
    plt.title("Figure 1. Testing and Training Error Graphs with degree dependency")
    plt.xlabel("Degree")
    plt.ylabel("Error")
    plt.legend(["Training Error","Testing Error"])
    plt.savefig('/home/devbhura/fall_2021/ML/HW1/hw1-knn-and-regression-devbhura/src/figures/ErrorvsDegreeA.png')



    # 
    # SplitB 
    # 

    degree = np.arange(10)
    error_train = []
    error_test = []
    for i in degree:

        p = PolynomialRegression(i)
        p.fit(x_train_B, y_train_B)
        y_hat = p.predict(x_train_B)
        mse_train = mean_squared_error(y_train_B,y_hat)
        error_train.extend([mse_train])
        
        plt.figure()
        plt.scatter(x_train_B,y_train_B)
        xs, ys = zip(*sorted(zip(x_train_B, y_hat)))
        plt.plot(xs,ys)

        # print(f"x_train={x_train_A}")
        # print(f"y_train={y_train_A}")
        # print(f"y_hat={y_hat}")

        p.fit(x_train_B, y_train_B)
        
        y_hat = p.predict(x_test_B)
        xs, ys = zip(*sorted(zip(x_test_B, y_hat)))
        plt.plot(xs,ys)
        plt.legend(["Polyfit train","Polyfit test"])
        plt.title(f"Polynomial Curves for degree = {i}")
        plt.savefig(f"/home/devbhura/fall_2021/ML/HW1/hw1-knn-and-regression-devbhura/src/figures/PolyfitDegB{i}.png")

        mse_test = mean_squared_error(y_test_B,y_hat)
        error_test.extend([mse_test])
       
    indextrain = error_train.index(min(error_train))
    indextest = error_test.index(min(error_test))

    plt.figure()
    plt.scatter(x_train_B,y_train_B)
    i_train = degree[indextrain]
    i_test = degree[indextest]

    p = PolynomialRegression(i_train)
    p.fit(x_train_B, y_train_B)
    y_hat = p.predict(x_train_B)

    xs, ys = zip(*sorted(zip(x_train_B, y_hat)))
    plt.plot(xs,ys)

    p = PolynomialRegression(i_test)
    p.fit(x_train_B, y_train_B)
    y_hat = p.predict(x_test_B)

    xs, ys = zip(*sorted(zip(x_test_B, y_hat)))
    plt.plot(xs,ys)

    plt.legend([f"Lowest Training Error deg={i_train} ",f"Lowest Test Error deg={i_test} "])
    plt.title(f"Polyfit with lowest error")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f"/home/devbhura/fall_2021/ML/HW1/hw1-knn-and-regression-devbhura/src/figures/PolyfitMinB.png")

    plt.figure()
    plt.plot(degree,np.log(error_train))
    plt.plot(degree,np.log(error_test))
    plt.title("Figure 1. Testing and Training Error Graphs with degree dependency")
    plt.xlabel("Degree")
    plt.ylabel("Error")
    plt.legend(["Training Error","Testing Error"])
    plt.savefig('/home/devbhura/fall_2021/ML/HW1/hw1-knn-and-regression-devbhura/src/figures/ErrorvsDegreeB.png')


    #
    # kNN SplitA 
    #
    x_train_A = x_train_A[:, None]  # expand dims
    y_train_A = y_train_A[:, None]
    x_test_A = x_test_A[:, None]
    k = [1, 3, 5, 7, 9]
    error_train = []
    error_test = []
    for i in k:

        knn = KNearestNeighbor(i, distance_measure='euclidean', aggregator='mean')
        knn.fit(x_train_A, y_train_A)
        y_hat = knn.predict(x_train_A)
        mse_train = mean_squared_error(y_train_A,y_hat)
        error_train.extend([mse_train])
        
        plt.figure()
        plt.scatter(x_train_A,y_train_A)
        xs, ys = zip(*sorted(zip(x_train_A, y_hat)))
        plt.plot(xs,ys)

        # print(f"x_train={x_train_A}")
        # print(f"y_train={y_train_A}")
        # print(f"y_hat={y_hat}")

        knn.fit(x_train_A, y_train_A)
        
        y_hat = knn.predict(x_test_A)
        xs, ys = zip(*sorted(zip(x_test_A, y_hat)))
        plt.plot(xs,ys)
        plt.legend(["Polyfit train","Polyfit test"])
        plt.title(f"kNN Polynomial Curves for k = {i}")
        plt.savefig(f"/home/devbhura/fall_2021/ML/HW1/hw1-knn-and-regression-devbhura/src/figures/PolyfitkNNSplitA{i}.png")

        mse_test = mean_squared_error(y_test_A,y_hat)
        error_test.extend([mse_test])
       
    indextrain = error_train.index(min(error_train))
    indextest = error_test.index(min(error_test))

    plt.figure()
    plt.scatter(x_train_A,y_train_A)
    i_train = k[indextrain]
    i_test = k[indextest]

    p = KNearestNeighbor(i_train)
    p.fit(x_train_A, y_train_A)
    y_hat = p.predict(x_train_A)

    xs, ys = zip(*sorted(zip(x_train_A, y_hat)))
    plt.plot(xs,ys)

    p = KNearestNeighbor(i_test)
    p.fit(x_train_A, y_train_A)
    y_hat = p.predict(x_test_A)

    xs, ys = zip(*sorted(zip(x_test_A, y_hat)))
    plt.plot(xs,ys)

    plt.legend([f"Lowest Training Error k={i_train} ",f"Lowest Test Error k={i_test} "])
    plt.title(f"kNN Polyfit with lowest error")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f"/home/devbhura/fall_2021/ML/HW1/hw1-knn-and-regression-devbhura/src/figures/PolyfitMinAkNN.png")

    plt.figure()
    plt.plot(k,error_train)
    plt.plot(k,error_test)
    plt.title("Figure 1. Testing and Training Error Graphs with degree dependency")
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.legend(["Training Error","Testing Error"])
    plt.savefig('/home/devbhura/fall_2021/ML/HW1/hw1-knn-and-regression-devbhura/src/figures/ErrorvskNNA.png')


    #
    # kNN SplitB
    #
    x_train_B = x_train_B[:, None]  # expand dims
    y_train_B = y_train_B[:, None]
    x_test_B = x_test_B[:, None]
    
    k = [1, 3, 5, 7, 9]
    error_train = []
    error_test = []
    for i in k:

        knn = KNearestNeighbor(i, distance_measure='euclidean', aggregator='mean')
        knn.fit(x_train_B, y_train_B)
        y_hat = knn.predict(x_train_B)
        mse_train = mean_squared_error(y_train_B,y_hat)
        error_train.extend([mse_train])
        
        plt.figure()
        plt.scatter(x_train_B, y_train_B)
        xs, ys = zip(*sorted(zip(x_train_B, y_hat)))
        plt.plot(xs,ys)

        # print(f"x_train={x_train_A}")
        # print(f"y_train={y_train_A}")
        # print(f"y_hat={y_hat}")

        knn.fit(x_train_B, y_train_B)
        
        y_hat = knn.predict(x_test_B)
        xs, ys = zip(*sorted(zip(x_test_B, y_hat)))
        plt.plot(xs,ys)
        plt.legend(["Polyfit train","Polyfit test"])
        plt.title(f"kNN Polynomial Curves for k = {i}")
        plt.savefig(f"/home/devbhura/fall_2021/ML/HW1/hw1-knn-and-regression-devbhura/src/figures/PolyfitkNNSplitB{i}.png")

        mse_test = mean_squared_error(y_test_B,y_hat)
        error_test.extend([mse_test])
       
    indextrain = error_train.index(min(error_train))
    indextest = error_test.index(min(error_test))

    plt.figure()
    plt.scatter(x_train_B,y_train_B)
    i_train = k[indextrain]
    i_test = k[indextest]

    p = KNearestNeighbor(i_train)
    p.fit(x_train_B, y_train_B)
    y_hat = p.predict(x_train_B)

    xs, ys = zip(*sorted(zip(x_train_B, y_hat)))
    plt.plot(xs,ys)

    p = KNearestNeighbor(i_test)
    p.fit(x_train_B, y_train_B)
    y_hat = p.predict(x_test_B)

    xs, ys = zip(*sorted(zip(x_test_B, y_hat)))
    plt.plot(xs,ys)

    plt.legend([f"Lowest Training Error k={i_train} ",f"Lowest Test Error k={i_test} "])
    plt.title(f"kNN Polyfit with lowest error")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f"/home/devbhura/fall_2021/ML/HW1/hw1-knn-and-regression-devbhura/src/figures/PolyfitMinBkNN.png")

    plt.figure()
    plt.plot(k,error_train)
    plt.plot(k,error_test)
    plt.title("Figure 1. Testing and Training Error Graphs with degree dependency")
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.legend(["Training Error","Testing Error"])
    plt.savefig('/home/devbhura/fall_2021/ML/HW1/hw1-knn-and-regression-devbhura/src/figures/ErrorvskNNB.png')



    return error_train, error_test


[error_train, error_test] = poly_reg()
degree = np.arange(10)



# print(f"target = {target}")