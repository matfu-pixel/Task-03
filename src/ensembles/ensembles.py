import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from time import time


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
        if 'random_state' in trees_parameters:
            self.random_state = trees_parameters['random_state']
        else:
            self.random_state = 34567

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
            history['time'] = []
            if X_val is not None:
                history['score_val'] = []
                history['best_pair'] = (0, 1e9)

        y_pred_train = np.zeros(X.shape[0])
        if trace and X_val is not None:
            y_pred_val = np.zeros(X_val.shape[0])

        self.feature_samples = []
        local_state = np.random.RandomState(self.random_state)
        start = time()
        for i in range(self.n_estimators):
            ind_row = local_state.choice(X.shape[0], X.shape[0])
            ind_col = local_state.choice(X.shape[1], self.feature_subsample_size, replace=False)
            self.feature_samples.append(ind_col)
            self.trees.append(DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state))
            self.trees[i].fit(X[ind_row][:, ind_col], y[ind_row])
            y_pred_train += self.trees[i].predict(X[:, ind_col])
            if trace:
                history['score_train'].append(mean_squared_error(y, y_pred_train / (i + 1), squared=False))
                history['time'].append(time() - start)
                if X_val is not None:
                    y_pred_val += self.trees[i].predict(X_val[:, ind_col])
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
        return np.mean([self.trees[i].predict(X[:, self.feature_samples[i]]) for i in range(len(self.trees))], axis=0)


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
        if 'random_state' in trees_parameters:
            self.random_state = trees_parameters['random_state']
        else:
            self.random_state = 34567

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
            history['time'] = []
            if X_val is not None:
                history['score_val'] = []
                history['best_pair'] = (0, 1e9)

        if self.feature_subsample_size is None:
            self.feature_subsample_size = X.shape[1] // 3

        self.y_train_mean = np.mean(y)
        y_pred_train = np.zeros(X.shape[0]) + self.y_train_mean
        if trace and X_val is not None:
            y_pred_val = np.zeros(X_val.shape[0]) + np.mean(y_val)

        start = time()
        self.feature_samples = []
        local_state = np.random.RandomState(self.random_state)
        for i in range(self.n_estimators):
            ind_row = local_state.choice(X.shape[0], X.shape[0])
            ind_col = local_state.choice(X.shape[1], self.feature_subsample_size, replace=False)
            self.feature_samples.append(ind_col)
            self.trees.append(DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state))
            self.trees[i].fit(X[ind_row][:, ind_col], y[ind_row] - y_pred_train[ind_row])
            y_pred_i = self.trees[i].predict(X[ind_row][:, ind_col])
            gamma = minimize_scalar(lambda x: mean_squared_error(
                y[ind_row], y_pred_train[ind_row] + x * y_pred_i, squared=False)).x
            self.weights.append(gamma)
            y_pred_train += self.learning_rate * gamma * self.trees[i].predict(X[:, ind_col])

            if trace:
                history['time'].append(time() - start)
                history['score_train'].append(mean_squared_error(y, y_pred_train, squared=False))
                if X_val is not None:
                    y_pred_val += self.learning_rate * gamma * self.trees[i].predict(X_val[:, ind_col])
                    history['score_val'].append(mean_squared_error(y_val, y_pred_val, squared=False))
                    value = mean_squared_error(y_val, y_pred_val, squared=False)
                    if  value < history['best_pair'][1]:
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
        return self.y_train_mean + self.learning_rate * (np.array(self.weights).reshape((len(self.weights), 1)) * np.array([self.trees[i].predict(X[:, self.feature_samples[i]]) for i in range(len(self.trees))])).sum(axis=0)
