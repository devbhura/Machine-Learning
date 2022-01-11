## Problems

## Code Implementation (8 points)
Pass test cases by implementing the functions in the `src` directory.

Your score for this section is defined by the autograder. If it says you got an 80/100, you get all 8 points here.

## Free-response Questions (12 points)

----
### The combinatorics of finding the truth (1 point)
1. The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) consists of 60000 images.  Each image is labeled with one of 10 class labels. There are 50000 training images and 10000 test images. The concept we're trying to learn is a function *C*(*x*) that maps every example in CFAR-10 to one of the true class labels. A hypothesis *h*(*x*) is also a function that maps every image in CIFAR-10 to one of the labels.
  - How many total hypotheses are possible? 
  - If you randomly draw a hypothesis, what is the probability it will label the training data perfectly? 
  - If you have a hypothesis that labels the training data perfectly, what is the probability it also labels the test data perfectly? 
 
----
### Making a metric (1 point)
2. I’d like to make a nearest neighbor, user-based collaborative filter for movie viewers using the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/). This presents a number of design choices. 
  - How are user-preferences represented in this data? User preference  is captured by the movie ratings from 1-5
  - What distance metric should I choose? Explain your reasoning.
  Manhattan Distance metric as the data set is discrete and not continuous
  - What should I do to handle missing values? Explain your reasoning.
  Remove the the data point. In this large data set, one should get enough data even if one removes missing values. This prevents any bias occuring from giving that data point an arbritary number especially in the case of a discrete data set
  - What issues or pitfalls might your approach have?
You will have less data to deal with and therefore might have less accurate models
----
### Thinking about models (4 points)
Assume you want to learn a function *f*(*x*) = *y*. The training data has *n* examples. Each example is encoded as a pair <*x*,*y*>, where *x* is a *d* dimensional input vector of real numbers and *y* is a scalar. 

You have a polynomial regressor and a kNN model, Assume both models uses a typical L-norm (e.g. Euclidean) as the distance metric. Assume a straightforward implementation of both models, like the ones discussed in class. Answer the following questions. Use [Big O notation](https://web.mit.edu/16.070/www/lecture/big_o.pdf) in your answers. Explain your reasoning.

3. About the polynomial regressor:
  - Show the time complexity of training a model, in terms of *d* and *n*.
  - Show, in terms of *d* and *n*,  the time complexity of of using a trained model to predict the *y* for a new *x*. 
  - Show the space complexity of a trained model, in terms of *d* and *n*.

4. About the kNN model:
  - Show the time complexity of training a model, in terms of *d* and *n*.
  - Show, in terms of *d* and *n*,  the time complexity of of using a trained model to predict the *y* for a new *x*. 
  - Show the space complexity of a trained model, in terms of *d* and *n*.

5. You have the following two data points in your training data < *x* = 0, *y* = 0 >, < *x* = 1, *y* = 1 >. You have a query *x* = 100 for which you must predict *y*. Assume a Euclidean norm. Answer the following questions.
  - What would a linear regressor (i.e. polynomial degree 1) predict for *y*?   
  - What would a kNN regressor where *k*=1 predict for *y*?
  - In your own words, describe the inductive bias of both models.

6. What is the primary influence on the answers produced by a trained model when presented data far outside its training distribution? Given this, should we use models outside their training distribution? Why or why not?

----
### Running polynomial regression  (3 points)

Use the function you implemented `generate_regression_data` to generate 100 points of data (set `amount_of_noise` = 0.1) from a polynomial function of degree 4. From this, select two test/train splits:

Split A: Randomly select 10 points as your training data. The rest will be your testing data. 

Split B: Randomly select 50 points as your training data. The rest will be your testing data. 

_Hint: save these splits, we'll use them again for the kNN classifier_

7. Run your implementation of `PolynomialRegression` on Split A, once for each degree of polynomial from 0 to 9. Then create the following two graphs:
    - A graph that shows the error of your regression as a function of degree. Make the horizontal dimension be the degree of the polynomial used. Make the vertical dimension the error. Label your axes. Put two lines on this plot, one that shows TRAINING error, and one that shows TESTING error. Label which is testing and which is training error.
  _Hint: Sometimes you don't see the error for some polynomial degree because another one has huge error, making the scale too large to see the error for the other (also bad) polynomial. If you might have this situation, try taking the log of the error before plotting it._
  
    - A second graph that shows a scatterplot of your training data. Overlay this scatterplot with plots of the following polynomial curves.
      - The polynomial with lowest testing error, Label this curve with its degree and "lowest testing error".
      - The polynomial with lowest training error. Label this curve with its degree and "lowest training error".

8. Given your plots from the previous question, describe the relationship between the degree of the polynomial used to fit the data and the training/testing error. At what degree of polynomial do you think you started overfitting the data?

9. Repeat everything you did in the previous question, this time with Split B. Compare your results on the two splits. How did increasing the number of training examples affect things? Suppose we trained these models on 1,000,000 samples instead of 100; would you expect a degree 2 or degree 9 polynomial to have lower TESTING error on this data? Explain your reasoning.

----
#### Running a kNN model (3 points)
10. Run your implementation of `KNearestNeighbor` on Split A, once for each value of *k* from the set {1, 3, 5, 7, 9}. Do the exact thing again for Split B. For all runs set *distance_measure* ='euclidean', and *aggregator*='mean'. Then do the create the following two graphs **for each split** (resulting in a total of 4 graphs):
    - A graph that shows the error of your regression as a function of *k*. Make the horizontal dimension be *k*. Make the vertical dimension the error. Label your axes. Put two lines on this plot, one that shows TRAINING error, and one that shows TESTING error. Label which is testing and which is training error.
  
    - A second graph that shows a scatterplot of your training data. Overlay this scatterplot with a plots of the value of *k* with lowest testing error, Label this curve with its value for *k* and "lowest testing error".

11. We'll now visualize examples of good classifiers:
  * Run your implementation of kNN on the data in `clean-spiral.json`. Try *k* = 1, *k* = 3, *k* = 5. Try both Manhattan and Euclidean distance. Try mean, median and mode as your aggregator. Then, pick the best combination of these variables, tell us which combination that was, and show us the scatter plot of the data, overlaid with the decision surface. Here, we define "best" as "gives a clean-looking spiral decision surface that correctly classifies the most data points."

  *  Do everything you did for `clean-spiral.json` again, but this time for the data in `noisy-linear.json`. This data is noisy...so correctly classifying ALL the data might not be the best measure. This time "best" will mean "gives a clean-looking linear-ish decision surface that correctly classifies almost all the data." (On that plot, don't forget to tell us which combination of variables you used)  

12. Given what you learned from questions 10 and 11, describe the trade-offs in your choice of *k*, aggregator and distance metric. When would you pick a small k? When would you pick a large one? What effect did you see from different aggregators?


_Aside: you can upload the json files containing the datasets to http://ml-playground.com/ to see how different learning algorithms work on each dataset. This is not required for points, and you're definitely not allowed to submit scatter plots built by ml-playground, but it might be illuminating and useful for checking your work._
