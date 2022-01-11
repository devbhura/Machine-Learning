import os
import random

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


def build_causal_dataset():
    """
    Create a dataset containing Income, Credit, Group, and Loan variables.
    You do not need to run or edit this function.
    """
    random.seed(42)
    np.random.seed(42)

    N = 100
    subgroup = np.random.binomial(n=1, p=0.3, size=[N, 1])

    credit_mean = 120 - 20 * subgroup
    credit = np.random.normal(loc=credit_mean, scale=20 - 10 * subgroup, size=[N, 1])

    loan_prob = 0.5 + 0.4 * logistic(credit, 100, 200) - 0.5 * subgroup
    loan_prob = np.clip(loan_prob, 0, 1)
    loan = np.random.binomial(n=1, p=loan_prob, size=[N, 1])

    income_mean = 80 + 80 * subgroup - 0.3 * credit + 20 * loan
    income = np.random.normal(loc=income_mean, scale=20 + 20 * loan + 20 * subgroup, size=[N, 1])

    data = pd.DataFrame(np.concatenate([income, credit, subgroup, loan], axis=1),
                        columns=["I", "C", "G", "L"])

    data.to_csv("data/causal_data.csv", index=False)


def estimator_one(data, name="1st"):
    """
    Estimate the causal effect of Loan on Income using Estimator 1
    Which Formula does this correspond to?
    """
    X_cols = ["L", "C", "G"]
    X = data[X_cols].to_numpy()
    y = data["I"].to_numpy()

    model = LinearRegression()
    model.fit(X, y)

    parameters = [f"{a:.3f}*{b}" for a, b in zip(model.coef_, X_cols)]
    intercept = model.intercept_
    sign = "+" if intercept >= 0 else "-"

    print(f"{name} Estimator: {' + '.join(parameters)} {sign} {np.abs(model.intercept_):.3f}")

    estimate = 0
    for row in X:
        row[X_cols.index("L")] = 1
        estimate += model.predict(row.reshape(1, -1))[0]

        row[X_cols.index("L")] = 0
        estimate -= model.predict(row.reshape(1, -1))[0]

    return estimate / data.shape[0]


def estimator_two(data, name="2nd"):
    """
    Estimate the causal effect of Loan on Income using Estimator 2
    Which Formula does this correspond to?
    """
    X_cols = ["L", "C"]
    X = data[X_cols].to_numpy()
    y = data["I"].to_numpy()

    model = LinearRegression()
    model.fit(X, y)
    parameters = [f"{a:.3f}*{b}" for a, b in zip(model.coef_, X_cols)]
    intercept = model.intercept_
    sign = "+" if intercept >= 0 else "-"

    print(f"{name} Estimator: {' + '.join(parameters)} {sign} {np.abs(model.intercept_):.3f}")

    estimate = 0
    for row in X:
        row[X_cols.index("L")] = 1
        estimate += model.predict(row.reshape(1, -1))[0]

        row[X_cols.index("L")] = 0
        estimate -= model.predict(row.reshape(1, -1))[0]

    return estimate / data.shape[0]


def free_response_six():
    data = pd.read_csv("data/causal_data.csv")

    estimate_one = estimator_one(data, name="1st")
    print(f"1st Estimate: {estimate_one:.2f}")

    estimate_two = estimator_two(data, name="2nd")
    print(f"2nd Estimate: {estimate_two:.2f}")


if __name__ == "__main__":
    free_response_six()
