import numpy as np

class Perceptron(object):
    # constructor
    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    # training algorithm
    # X: a numpy 2d array of samples and features with a shape (number of samples, number of features
    # y: a numpy array of target labels with shape (number of samples)
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        # get a normal distribution
        #   loc = mean of the distribution (center point)
        #   scale = standard deviation of the distribution (the spread / width)
        #   size = number of random numbers returned, default is 1
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        # iterate a set number of times to train on the data passed in
        for _ in range(self.n_iter):
            errors = 0

            # iterate the samples and targets
            # since they are the same size we fully iterate both lists
            # xi = is input from sample
            # target = correct answer
            for xi, target in zip(X, y):
                # make a prediction for the xi input vector and calculate the error
                update = self.eta * (target - self.predict(xi))
                # update all weights from element 1 onward with the error times the input
                self.w[1:] += update * xi
                # update weight 0 with only the error (the bias)
                self.w[0] += update
                # track our errors for debugging
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def predict(self, X):
        # returns x or y based on condition
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    def net_input(self, X):
        # dot product of input vector and weight vector + w0 (the bias)
        return np.dot(X, self.w_[1:]) + self.w_[0]