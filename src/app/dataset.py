import numpy as np


class Dataset:
    def __init__(self, target_name, data_train, data_val):
        self.target_name = target_name
        self.data_train = data_train
        self.data_val = data_val
        if not Dataset.check_data(target_name, data_train, data_val):
            raise TypeError

    def check_data(target_name, data_train, data_val):
        num_cols_train = data_train.select_dtypes(include=np.number).columns.tolist()

        if target_name in data_train.columns.values and \
            len(num_cols_train) == data_train.shape[1]:
                if data_val is not None:
                    num_cols_val = data_val.select_dtypes(include=np.number).columns.tolist()
                    if num_cols_train == num_cols_val and \
                        len(num_cols_val) == data_val.shape[1]:
                        return True
                    else:
                        return False
                else:
                    return True
        else:
            return False
