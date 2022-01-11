import numpy as np


def softmax(x, axis):
    """
    Implements a *stabilized* softmax along the correct index
    https://www.deeplearningbook.org/contents/numerical.html

    Do not use scipy to implement this function!
    """
    x = np.atleast_2d(x)
    max = np.max(x,axis=axis)
    x_sub = x - np.array([max,max]).T
    e = np.exp(x_sub)
    summation = np.sum(e,axis=axis)
    
    e[:,0] = e[:,0]/summation
    e[:,1] = e[:,1]/summation

    return e
