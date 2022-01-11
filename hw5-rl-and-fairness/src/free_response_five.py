import csv
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', family="serif", size=14)



def metrics(preds, labels):
    """
    Compute the four metrics (PPV, NPV, FPR, and FNR) which
        we use to evaluate the fairness of a classifier.
        These metrics are defined in the lecture slides and
        the assigned reading (see Free Response Question 4).

        There are no test cases for this function, but it will
        be used in `train_logistic_regression` below and you
        will need its output to answer parts 5b through 5e.

        You may use sklearn.metrics.confusion_matrix or your
        implementation from Homework 2.

    args:
        preds: the labels predicted by the classifier
        labels: the true labels against which to compare

    returns:
        a string containing the four metrics: PPV, NPV, FPR, and FNR
    """
    tp = 0 
    tn = 0 
    fp = 0 
    fn = 0

    predictions = preds
    actual = labels
    # if predictions.shape[0] != actual.shape[0]:
    #     raise ValueError("predictions and actual must be the same length!")
    # else:
    # for i in range(actual.shape[0]):
    #     if actual[i] == predictions[i]:
    #         if predictions[i] == True:
    #             tp += 1
    #         else:
    #             tn +=1 

    #     else:
    #         if predictions[i] == True:
    #             fp += 1
    #         else:
    #             fn += 1

    tn, fp, fn, tp = confusion_matrix(actual,predictions).ravel()

    a = tp
    b = fn
    c = fp
    d = tn
    ppv = a/(a+c)
    npv = d/(b+d)
    fnr = b/(a+b)
    fpr = c/(c+d)

    return f"PPV: {ppv:.2f} NPV: {npv:.2f} FPR: {fpr:.2f} FNR: {fnr:.2f}"


def train_logistic_regression(X, y, group, where=None, name=''):
    """
    Train a logistic regression model and prints out performance metrics.
        You should not need to modify this code to answer the questions.
        The model is defined as g(w_1 * x_1 + ... + w_k * x_k + b)
        where g is the logistic function.

    args:
        X: the features
        y: the label
        group: a column indicating which individuals belong to the protected subgroup
        where: if not None, only train on individuals where this array is True
        name: a helper string for printouts
    returns:
        logreg: the trained logistic regression model
    """

    if where is not None:
        X = X[where]
        y = y[where]
        group = group[where]

    logreg = LogisticRegression()
    logreg.fit(X, y)
    preds = logreg.predict(X)

    group0 = group == 0
    group1 = group == 1

    # Print out the model's coefficients and intercept.
    parameters = [f"{a:.3f}*{b}" for a, b in zip(logreg.coef_[0], X.columns)]
    intercept = logreg.intercept_[0]
    sign = "+" if intercept >= 0 else "-"
    print(f"{name} Model: g({' + '.join(parameters)} {sign} {np.abs(logreg.intercept_[0]):.3f})")

    # Print overall performance
    print(f"{name} Overall {metrics(preds, y)}")

    # Unless restricted to a specific group, also print the performance within each group
    if where is None:
      print(f"{name} Group 0 {metrics(preds[group0], y[group0])}")
      print(f"{name} Group 1 {metrics(preds[group1], y[group1])}")
    print()

    return logreg


def build_figure(data):
    '''
    Build a plot for the four visualizations
    '''
    nrow = 2
    ncol = 2
    fig, axes = plt.subplots(
        nrow, nrow, sharex=True, sharey=True, figsize=(8, 8))
    axes = axes.reshape(-1).tolist()

    # Create custom legend with shape and color
    def handle_plot(m, c):
        return axes[0].plot([], [], marker=m, color=c, ls="none")[0]

    cmap = plt.get_cmap("Set1")
    handle_args = [("s", cmap(0.0)), ("s", cmap(1.0)), ("x", "k"), ("o", "k")]
    handles = [handle_plot(*x) for x in handle_args]
    labels = ["No Loan", "Loan", "Group 0", "Group 1",]
    fig.legend(handles, labels, ncol=4, loc="upper center", framealpha=1.0)

    for i, axis in enumerate(axes):
        if i % 2 == 0:
            axis.set_ylabel("Credit")
        if i >= (nrow * ncol // 2):
            axis.set_xlabel("Income")

    return (fig, axes)


def visualize_data(data, ax, name=''):
    '''
    Visualize the data for this problem
    '''
    xmin, xmax = np.percentile(data["I"], [0, 100])
    ymin, ymax = np.percentile(data["C"], [0, 100])
    ax.set_xlim(xmin - 10, xmax + 10)
    ax.set_ylim(ymin - 10, ymax + 10)
    x_arr = np.array([xmin - 10, xmax + 10])

    group0 = data["G"] == 0
    group1 = data["G"] == 1

    ax.scatter(
       x=data["I"][group0], y=data["C"][group0], cmap="Set1", c=data["L"][group0], marker="x")
    ax.scatter(x=data["I"][group1], y=data["C"][group1], cmap="Set1", c=data["L"][group1], marker="o")


    if len(name) > 0:
        ax.set_title(name)


def visualize_logistic_regression(logreg, ax):
    '''
    Visualize the logistic regression model.
    For these problems, the positive side of the decision boundary is
        shaded grey, and the negative side is shaded red.
    '''

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x_arr = np.array([xmin, xmax])

    b = logreg.intercept_[0]
    w1, w2 = logreg.coef_.T[:2]

    y_arr = (-w1 / w2) * x_arr - (b / w2)

    ax.plot(x_arr, y_arr, 'k', lw=1, ls='--')
    ax.fill_between(x_arr, y_arr, ymin - 20, color="tab:red", alpha=0.2)
    ax.fill_between(x_arr, y_arr, ymax + 20, color="tab:grey", alpha=0.2)


def part_b(data, ax):
    """
    Train a logistic regression to predict Loan using Income and Credit as features
    """
    name = "b."
    logreg = train_logistic_regression(data[["I", "C"]], data["L"], data["G"], name=name)
    visualize_data(data, ax, name)
    visualize_logistic_regression(logreg, ax)
    return logreg


def part_c(data, ax):
    """
    Train a logistic regression to predict Loan using Income, Credit, and Group as features
    """
    name = "c."
    logreg = train_logistic_regression(data[["I", "C", "G"]], data["L"], data["G"], name=name)
    visualize_data(data, ax, name)
    visualize_logistic_regression(logreg, ax)
    return logreg


def part_d(data, ax, group):
    """
    Train a logistic regression to predict Loan, using *only* on individuals where Group = {group}
    """
    name = f"d. G={group}"
    where = data["G"] == group
    logreg = train_logistic_regression(data[["I", "C"]], data["L"], data["G"], where=where, name=name)
    visualize_data(data, ax, name)
    visualize_logistic_regression(logreg, ax)
    return logreg


def free_response_five():
    data = pd.read_csv("data/fairness_data.csv")

    fig, axes = build_figure(data)

    part_b(data, axes[0])
    part_c(data, axes[1])
    part_d(data, axes[2], group=0)
    part_d(data, axes[3], group=1)

    plt.show()
    plt.close("all")


if __name__ == "__main__":
  free_response_five()
