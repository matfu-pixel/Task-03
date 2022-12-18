from ensembles import RandomForestMSE, GradientBoostingMSE
import pandas as pd


class Model:
    def __init__(self, type, n_estimators, feature_subsample_size, max_depth, learning_rate):
        self.type = type
        self.n_estimators = n_estimators
        self.feature_subsample_size = feature_subsample_size
        self.max_depth = max_depth
        self.learning_rate = learning_rate

        params = ['Type', 'n_estimators', 'feature_subsample_size', 'max_depth']
        values = [type, n_estimators, feature_subsample_size, max_depth]
        if type == 'GradientBoosting':
            values.append(learning_rate)
            params.append('learning_rate')
        self.info = pd.DataFrame({
            'Parameter': params,
            'Value': values
        })


    def fit(self, data_train, data_val, target_name):
        if self.type == 'RandomForest':
            self.model = RandomForestMSE(n_estimators=self.n_estimators,
                feature_subsample_size=self.feature_subsample_size,
                max_depth=self.max_depth
            )
        else:
            self.model = GradientBoostingMSE(n_estimators=self.n_estimators,
                feature_subsample_size=self.feature_subsample_size,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate
            )

        x_val = None if data_val is None else data_val.drop([target_name], axis=1).values
        y_val = None if data_val is None else data_val[target_name].values
        self.history = self.model.fit(data_train.drop([target_name], axis=1).values,
                                      data_train[target_name].values,
                                      x_val,
                                      y_val,
                                      trace=True)
        return self.history

    def predict(self, data_test):
        return self.model.predict(data_test.values)