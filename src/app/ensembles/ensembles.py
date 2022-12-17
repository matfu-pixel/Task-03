import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


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
        self.max_depth = max_depth
        self.feature_subsample_size=feature_subsample_size
        self.trees_parameters = trees_parameters
        self.trees = []

    def fit(self, X, y, X_val=None, y_val=None, trace=False):
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
        if self.feature_subsample_size is None:
            self.feature_subsample_size = X.shape[1] // 3

        if trace:
            history = dict()
            history['score_train'] = []
            if X_val is not None:
                history['score_val'] = []
                history['best_pair'] = (0, 1e9)

        y_pred_train = np.zeros(X.shape[0])
        if trace and X_val is not None:
            y_pred_val = np.zeros(X_val.shape[0])

        for i in range(self.n_estimators):
            ind = np.random.choice(X.shape[0], X.shape[0])
            self.trees.append(DecisionTreeRegressor(max_depth=self.max_depth,
                                                    max_features=self.feature_subsample_size))
            self.trees[i].fit(X[ind], y[ind])
            y_pred_train += self.trees[i].predict(X)
            if trace:
                history['score_train'].append(mean_squared_error(y, y_pred_train / (i + 1), squared=False))
                if X_val is not None:
                    y_pred_val += self.trees[i].predict(X_val)
                    value = mean_squared_error(y_val, y_pred_val / (i + 1), squared=False)
                    history['score_val'].append(value)
                    if value < history['best_pair'][1]:
                        history['best_pair'] = (i, value)

        if trace:
            return history


    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        return np.mean([tree.predict(X) for tree in self.trees], axis=0)


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
        self.max_depth = max_depth
        self.feature_subsample_size= feature_subsample_size
        self.trees_parameters = trees_parameters
        self.weights = []
        self.trees = []

    def fit(self, X, y, X_val=None, y_val=None, trace=False):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        """
        if trace:
            history = dict()
            history['score_train'] = []
            if X_val is not None:
                history['score_val'] = []

        self.y_train_mean = np.mean(y)
        y_pred_train = np.zeros(X.shape[0]) + self.y_train_mean
        if trace and X_val is not None:
            y_pred_val = np.zeros(X_val.shape[0]) + np.mean(y_val)

        for i in range(self.n_estimators):
            ind = np.random.choice(X.shape[0], X.shape[0])
            self.trees.append(DecisionTreeRegressor(max_depth=self.max_depth,
                                                    max_features=self.feature_subsample_size))
            self.trees[i].fit(X[ind], y[ind] - y_pred_train[ind])
            y_pred_i = self.trees[i].predict(X[ind])
            gamma = minimize_scalar(lambda x: mean_squared_error(
                y[ind], y_pred_train[ind] + x * y_pred_i, squared=False)).x
            self.weights.append(gamma)
            y_pred_train += self.learning_rate * gamma * self.trees[i].predict(X)

            if trace:
                history['score_train'].append(mean_squared_error(y, y_pred_train, squared=False))
                if X_val is not None:
                    y_pred_val += self.learning_rate * gamma * self.trees[i].predict(X_val)
                    history['score_val'].append(mean_squared_error(y_val, y_pred_val, squared=False))

        if trace:
            return history

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        return self.y_train_mean + self.learning_rate * (np.array(self.weights).reshape((len(self.weights), 1)) * np.array([tree.predict(X) for tree in self.trees])).sum(axis=0)
