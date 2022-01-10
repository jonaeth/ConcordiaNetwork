import numpy as np
from sklearn.decomposition import NMF
import itertools


class DataTransformation:
    def __init__(self, df_users, df_items, df_ratings, categories_transformed=False):
        self.df_items = df_items
        self.df_ratings = df_ratings
        self.df_users = df_users
        self.nr_of_items = len(df_items)
        self.nr_of_users = len(df_users)
        self.rating_matrix = self.build_rating_matrix()
        self.user_latent_space, self.item_latent_space = self.build_mf_latent_vectors()
        if not categories_transformed:
            self.transform_category_vectors()

        print(f'Number of items: {self.nr_of_items}; Number of users: {self.nr_of_users}')

    def build_rating_matrix(self):
        rating_matrix_data = np.zeros((self.nr_of_items, self.nr_of_users))
        for item_id, user_id, rating in self.df_ratings[['item_id', 'user_id', 'rating']].values:
            rating_matrix_data[int(item_id), int(user_id)] = int(rating)
        return rating_matrix_data

    def build_mf_latent_vectors(self):
        print('Building latent sapces')
        model = NMF(n_components=8, init='random', random_state=0)
        user_latent_space = model.fit_transform(self.rating_matrix.T)
        item_latent_space = model.components_
        return user_latent_space, item_latent_space

    def transform_category_vectors(self):
        categories = set(list(itertools.chain.from_iterable(self.df_items.categories.values)))
        category_map = {category: i for i, category in enumerate(
            list(set(itertools.chain.from_iterable([category.strip().split(', ') for category in categories]))))}
        self.df_items['categories_binary'] = self.df_items['categories'].apply(
            lambda x: list(itertools.chain.from_iterable([category.strip().split(', ') for category in x])))

        def get_category_vector(category_list):
            vector = np.zeros(len(category_map))
            for category in category_list:
                vector[category_map[category]] = 1
            return list(vector)

        self.df_items['categories_binary'] = self.df_items['categories_binary'].apply(lambda x: get_category_vector(x))
