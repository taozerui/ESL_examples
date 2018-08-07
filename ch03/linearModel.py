import numpy as np
from scipy.linalg import inv

class Linear(object):
    """
    OLS solution for linear regression
    input: x, y
    output:
        coef_
        intercept_
    method:
        fit(x, y)
        predict(x)
    """
    def __init__(self):
        self.intercept_ = 0
        self.coef_ = 0

    def predict(self, x_test):
        y_hat = self.intercept_ + np.dot(x_test, self.coef_)
        return y_hat

class LeastSquare(Linear):
    def __init__(self):
        super(LeastSquare, self).__init__()

    def fit(self, x, y):
        try:
            n, p = x.shape
        except:
            n = x.shape
            x = x.reshape(-1, 1)
        y = y.reshape(n)
        alpha = np.mean(y)
        y = y - alpha
        beta = np.dot(inv(np.dot(x.T, x)),
                np.dot(x.T, y))
        self.coef_ = beta
        self.intercept_ = alpha

class Ridge(Linear):
    def __init__(self, alpha):
        super(Ridge, self).__init__()
        self.alpha = alpha

    def fit(self, x, y):
        n, p = x.shape
        y = y.reshape(n)
        alpha = np.mean(y)
        y = y - alpha
        beta = np.dot(inv(np.dot(x.T, x) + alpha * np.eye(p)),
                np.dot(x.T, y))
        self.coef_ = beta
        self.intercept_ = alpha

class Lasso(Linear):
    pass
