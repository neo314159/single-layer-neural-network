import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Logistic_Regression:
    """
    A Logistic Regression model, as a single-layer perceptron.
    """

    def __init__(self, iterations=10000, tolerance=0.00001, learning_rate=0.00001):
        self.iter = iterations
        self.tol = tolerance
        self.lr = learning_rate
        self._w = 0
        self._b = 0

    def sigmoid(self, z):
        """
        Parameters:
        -----------
        z: A scalar or numpy array of any size.
        Returns:
        --------
        s: sigmoid(z)
        """
        s = 1 / (1 + np.exp(-z))

        return s

    def initialize_parameters(self, n):
        """
        Parameters:
        -----------
        n: size of w vector (number of features).
        Returns:
        --------
        w -- initialized vector of shape (1, n)
        b -- initialized scalar (corresponds to the bias)
        """
        w = np.zeros((1, n))
        b = 0

        return w, b

    def fit(self, X, Y):
        """
        Parameters:
        ----------
        X: numpy.ndarray
            Training examples of input data of size (m, n)
        Y: numpy.ndarray
            The binary targets with the associated ground truths, of size (m,1)
        Returns:
        --------
        None
        """

        X = X.T  # transpose values, X becomes (n,m) matrix
        Y = Y.T  # transpose values, Y becomes (1,m) matrix

        m = X.shape[1]
        n = X.shape[0]

        w, b = self.initialize_parameters(n)

        for i in range(self.iter):
            # Forward Propagation
            A = self.sigmoid(np.dot(w, X) + b)  # compute activation
            cost = np.sum(((- np.log(A)) * Y + (-np.log(1 - A)) * (1 - Y))) / m  # compute cost (Log-Loss-Cross-Entropy)

            # # # Backward Propagation # # #
            # Here we need to adjust the weights backward
            # Given that A = 1/( 1 + exp(-Z) ) and Z = Weight Matrix (w) * Input Matrix (X) + B
            # and Cost = - [log(A) * Y + log(1 - A) * (1 - Y)]
            # The derivative of the cost function with respect to A is [ -Y/A + (1 - Y)/(1 - A) ]
            # While the derivative of A with respect to Z is [ A(1-A) ]

            # dz = [A - Y] is the result of the multiplication of the derivative of the cost function
            # with respect to A times the derivative of A with respect to Z,
            # where Z = w * X + B
            # The computation is dz = [ -Y/A + (1 - Y)/(1 - A) ] * [ A(1-A) ] = A - Y
            # In other words, we apply the chain rule to calculate the sensitivity of the cost function
            # to small changes in the weights
            dz = A - Y

            db = np.sum(dz) / m

            dw = np.dot(dz, X.T) / m
            # Update parameters
            w = w - self.lr * dw
            b = b - self.lr * db

            # # Print cost every 100 training iterations
            if i % 100 == 0:
                logger.info(f"Cost after iteration {i}: {cost}")

        self._w = w
        self._b = b

    def predict(self, X):
        """
        Parameters:
        ----------
        X: numpy.ndarray
            Test examples of input data of size (m, n)
        Returns:
        -------
        predictions: numpy.ndarray
            Predicted binary classes for each test examples
        """

        X = X.T  # transpose values, X becomes (n,m) matrix

        # Forward Propagation
        A = self.sigmoid(np.dot(self._w, X) + self._b)  # compute activation

        predictions = (A >= 0.5)[0]

        return predictions
