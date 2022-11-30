import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor


class RandomForestMSE:
    def __init__(
        self, n_estimators, max_depth=None, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.trees = \
        [DecisionTreeRegressor(max_depth=max_depth,
                               max_features=feature_subsample_size)
         for _ in range(n_estimators)]

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        X_val : numpy ndarray
            Array of size n_val_objects, n_features
        y_val : numpy ndarray
            Array of size n_val_objects
        """
        for i in range(self.n_estimators):
            self.trees[i].fit(X, y)
        if X_val is not None:
            pass

        return self


    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        sum = np.zeros(X.shape[0])
        for i in range(self.n_estimators):
            sum += self.trees[i].predict(X)

        return sum / self.n_estimators


class GradientBoostingMSE:
    def __init__(
        self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        learning_rate : float
            Use alpha * learning_rate instead of alpha
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.weights = np.zeros(n_estimators)
        self.weights[0] = 1 / learning_rate
        self.trees = \
        [DecisionTreeRegressor(max_depth=max_depth,
                               max_features=feature_subsample_size)
         for _ in range(n_estimators)]

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        """
        self.trees[0].fit(X, y)
        y_pred = self.trees[0].predict(X)

        for i in range(1, self.n_estimators):
            self.trees[i].fit(X, y - y_pred)
            self.weights[i] = 1 # !!!!!! тут реализовать наискорейший спуск черзе minimize_scalar
            y_pred += self.learning_rate * self.weights[i] * self.trees[i].predict(X)

        if X_val is not None:
            pass

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        sum = np.zeros(X.shape[0])
        for i in range(self.n_estimators):
            sum += self.learning_rate * self.weights[i] * self.trees[i].predict(X)

        return sum
