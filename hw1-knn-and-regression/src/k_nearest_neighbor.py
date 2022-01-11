import numpy as np 
from .distances import euclidean_distances, manhattan_distances



def mode(a, axis=0):
    """
    Copied from scipy.stats.mode. 
    https://github.com/scipy/scipy/blob/v1.7.1/scipy/stats/stats.py#L361-L451

    Return an array of the modal (most common) value in the passed array.
    If there is more than one such value, only the smallest is returned.
    The bin-count for the modal bins is also returned.
    Parameters
    ----------
    a : array_like
        n-dimensional array of which to find mode(s).
    axis : int or None, optional
        Axis along which to operate. Default is 0. If None, compute over
        the whole array `a`.
    """
    scores = np.unique(np.ravel(a))       # get ALL unique values
    testshape = list(a.shape)
    testshape[axis] = 1
    oldmostfreq = np.zeros(testshape)
    oldcounts = np.zeros(testshape)

    for score in scores:
        template = (a == score)
        counts = np.expand_dims(np.sum(template, axis),axis)
        mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
        oldcounts = np.maximum(counts, oldcounts)
        oldmostfreq = mostfrequent

    return mostfrequent, oldcounts

class KNearestNeighbor():    
    def __init__(self, n_neighbors, distance_measure='euclidean', aggregator='mode'):
        """
        K-Nearest Neighbor is a straightforward algorithm that can be highly
        effective. Training time is...well...is there any training? At test time, labels for
        new points are predicted by comparing them to the nearest neighbors in the
        training data.

        ```distance_measure``` lets you switch between which distance measure you will
        use to compare data points. The behavior is as follows:

        If 'euclidean', use euclidean_distances, if 'manhattan', use manhattan_distances.

        ```aggregator``` lets you alter how a label is predicted for a data point based 
        on its neighbors. If it's set to `mean`, it is the mean of the labels of the
        neighbors. If it's set to `mode`, it is the mode of the labels of the neighbors.
        If it is set to median, it is the median of the labels of the neighbors. If the
        number of dimensions returned in the label is more than 1, the aggregator is
        applied to each dimension independently. For example, if the labels of 3 
        closest neighbors are:
            [
                [1, 2, 3], 
                [2, 3, 4], 
                [3, 4, 5]
            ] 
        And the aggregator is 'mean', applied along each dimension, this will return for 
        that point:
            [
                [2, 3, 4]
            ]

        Hint: numpy has functions for computing the mean and median, but you can use the `mode`
              function for finding the mode. 

        Arguments:
            n_neighbors {int} -- Number of neighbors to use for prediction.
            distance_measure {str} -- Which distance measure to use. Can be one of
                'euclidean' or 'manhattan'. This is the distance measure
                that will be used to compare features to produce labels. 
            aggregator {str} -- How to aggregate a label across the `n_neighbors` nearest
                neighbors. Can be one of 'mode', 'mean', or 'median'.
        """
        self.n_neighbors = n_neighbors
        self.distance_measure =  distance_measure
        self.aggregator = aggregator
        


    def fit(self, features, targets):
        """Fit features, a numpy array of size (n_samples, n_features). For a KNN, this
        function should store the features and corresponding targets in class 
        variables that can be accessed in the `predict` function. Note that targets can
        be multidimensional! 
        
        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            targets {[type]} -- Target labels for each data point, shape of (n_samples, 
                n_dimensions).
        """

        self.features = features
        self.targets = targets
        # print(f"targets ={np.shape(self.targets)}")
        
        

    def predict(self, features, ignore_first=False):
        """Predict from features, a numpy array of size (n_samples, n_features) Use the
        training data to predict labels on the test features. For each testing sample, compare it
        to the training samples. Look at the self.n_neighbors closest samples to the 
        test sample by comparing their feature vectors. The label for the test sample
        is the determined by aggregating the K nearest neighbors in the training data.

        Note that when using KNN for imputation, the predicted labels are the imputed testing data
        and the shape is (n_samples, n_features).

        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            ignore_first {bool} -- If this is True, then we ignore the closest point
                when doing the aggregation. This is used for collaborative
                filtering, where the closest point is itself and thus is not a neighbor. 
                In this case, we would use 1:(n_neighbors + 1).

        Returns:
            labels {np.ndarray} -- Labels for each data point, of shape (n_samples,
                n_dimensions). This n_dimensions should be the same as n_dimensions of targets in fit function.
        """
        
        self.predict_features = features

        self.m, self.n = np.shape(self.predict_features)
        self.m_target, self.n_target = np.shape(self.targets)
        
        self.labels = np.zeros((self.m,self.n_target))
        # print(f"labels = {self.labels}")
        for i in range(self.m):

            x = [self.predict_features[i]]
            # print(f"x={np.shape(x)}")
            # print(f"features={np.shape(self.features)}")
            if self.distance_measure == 'euclidean':
            
                d = euclidean_distances(x,self.features)

            elif self.distance_measure == 'manhattan':
                d = manhattan_distances(x,self.features)

            # print(f"d={d}")
            index = d.argsort()[0]
            # print(f"index={np.shape(index)}")
            # print(f"index={index}")
            # print(f"target={self.targets}")
            if ignore_first == False:
                self.sorted_target = self.targets[index]
                # print(f"sorted_target={self.sorted_target}")
                self.sorted_target = self.sorted_target[:self.n_neighbors]
                # print(f"sorted_target={self.sorted_target}")
            elif ignore_first == True:
                self.sorted_target = self.targets[index]
                self.sorted_target = self.sorted_target[1:(self.n_neighbors+1)]  ###### MAYBE
            
            
            # self.sorted_target = self.sorted_target[0]

            # print(f"sorted_target={self.sorted_target}")
            if self.aggregator == 'mode':
                
                Y, oldcounts = mode(self.sorted_target)
                Y = Y[0]
                # print(f"mode = {Y}")

            elif self.aggregator == 'median':
                Y = np.median(self.sorted_target,axis = 0)
                # print(f"median={Y}")

            elif self.aggregator == 'mean':
                Y = np.mean(self.sorted_target,axis = 0)
                # print(f"mean={Y}")
            
            for j in range(len(Y)):
                self.labels[i][j] = Y[j]
            
            

        
        return self.labels


# if __name__ == "__main__":
    
#     features = np.array([
#         [-1, 1, 1, -1, 2],
#         [-1, 1, 1, -1, 1],
#         [-1, 2, 2, -1, 1],
#         [-1, 1, 1, -1, 1],
#         [-1, 1, 1, -1, 1]
#     ])

#     predict = np.array([
#         [-1, 1, 0, -1, 0],
#         [-1, 1, 1, -1, 0],
#         [-1, 0, 1, 0, 0],
#         [-1, 1, 1, -1, 1],
#         [-1, 1, 1, -1, 0]
#     ])
#     targets = np.array([
#         [1, 0, 1],
#         [1, 1, 5],
#         [3, 1, 1],
#         [1, 1, 2],
#         [5, 1, 1]
#     ])

#     knn = KNearestNeighbor(1, distance_measure='euclidean', aggregator='mean')
#     knn.fit(features, targets)

#     # predict and calculate accuracy
#     labels = knn.predict(predict)
#     print(f"predict={labels}")