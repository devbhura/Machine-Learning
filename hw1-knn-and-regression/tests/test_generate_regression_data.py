from src import generate_regression_data

def test_generate_regression_data():
    degrees = range(10)
    amounts = [10, 100, 1000, 10000]

    for degree in degrees:
        for amount in amounts:
            x, y = generate_regression_data(degree, amount)
            assert (len(x) == amount and len(y) == amount)
            assert (x.min() >= -1 and x.max() <= 1)
