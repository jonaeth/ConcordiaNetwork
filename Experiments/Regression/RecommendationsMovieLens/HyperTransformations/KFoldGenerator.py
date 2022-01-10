import numpy as np
from sklearn.model_selection import KFold


class KFoldGenerator:
    def __init__(self, rating_df):
        self.rating_df = rating_df

    def build_splits(self, nr_of_splits=5):
        kf = KFold(n_splits=nr_of_splits)
        rating_split = []
        for train_index, test_index in kf.split(self.rating_df):
            rating_learn_obs, rating_eval_obs = self.rating_df.iloc[train_index], self.rating_df.iloc[test_index]
            rating_split.append([rating_learn_obs, rating_eval_obs])
        return rating_split