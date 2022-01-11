import numpy as np 

def generate_regression_data(degree, N, amount_of_noise=1.0):
    """
    Generates data to test one-dimensional regression models. This function:

    1)  Generates explanatory variable x: a shape (N,) array that contains 
        Floats chosen at random between -1 and 1.

    2)  Creates a polynomial function f() of degree 'degree'. The polynomial's 
        Float coefficients are chosen uniformally at random between -10 and 10.

    3)  Generates response variable y: a shape (N,) array that contains f(x), 
        where the ith element of y is calculated by applying f() to the ith 
        element of x

    4)  Adds Gaussian noise n to y. Here mean(n) = 0 and standard deviation 
        (notated std(n)) is: std(n) = 'amount_of_noise' * std(y) and mean 0
        (Hint...use np.random.normal to generate this noise)

    Args:
        degree (int): degree of polynomial that relates the output x and y
        N (int): number of points to generate
        amount_of_noise (float): amount of random noise to add to the relationship 
            between x and y.
    Returns:
        x (np.ndarray): explanatory variable of size N, ranges between -1 and 1.
        y (np.ndarray): response variable of size N, which responds to x as a
                        polynomial of degree 'degree'.

    """
    x = np.random.uniform(-1,1,N)

    w = np.random.uniform(-10,10,degree)
    f = np.zeros(N)
    for i in range(degree):
        f+= w[i]*(x**i)

    y = f + np.random.normal(0,amount_of_noise*np.std(f),N)


    return x, y


