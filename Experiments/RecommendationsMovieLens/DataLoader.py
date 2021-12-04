import pandas as pd


class DataLoader:
    def __init__(self, split_type, path_to_data):
        self.split_type = split_type
        self.df_movies = self.__extract_movies_data(path_to_data + '/u.item')
        self.df_users = self.__extract_users_data(path_to_data + '/u.user')
        self.df_ratings = self.__extract_ratings(path_to_data + f'/split/u.data.{split_type}')
        self.__reindex_data_ids()
        self.__add_category_data()

    def __extract_ratings(self, path):
        df_ratings = pd.read_csv(path, sep='\t', names=['user_id', 'item_id', 'rating'])
        return df_ratings

    def __extract_movies_data(self, path):
        df_movies = pd.read_csv(path, sep='|', encoding='windows-1252', header=None,
                                names=['item_id', 'title', 'date', 'skip', 'imdb_url'] + [f'cat_{i}' for i in range(19)])
        return df_movies

    def __extract_users_data(self, path):
        df = pd.read_csv(path, sep='|', header=None, names=['user_id', 'age', 'gender', 'profession', 'zip'])
        return df

    def __add_category_data(self):
        self.df_users['categories_binary'] = pd.get_dummies(self.df_users['profession']).values.tolist()
        self.df_movies['categories_binary'] = self.df_movies[[f'cat_{i}' for i in range(19)]].values.tolist()

    def __reindex_data_ids(self):
        movies_new_indeces_map = {id: i for i, id in enumerate(self.df_movies['item_id'])}
        users_new_indeces_map = {id: i for i, id in enumerate(self.df_users['user_id'])}
        self.df_ratings['user_id'] = self.df_ratings['user_id'].apply(lambda x: users_new_indeces_map[x])
        self.df_ratings['item_id'] = self.df_ratings['item_id'].apply(lambda x: movies_new_indeces_map[x])
        self.df_movies['item_id'] = self.df_movies['item_id'].apply(lambda x: movies_new_indeces_map[x])
        self.df_users['user_id'] = self.df_users['user_id'].apply(lambda x: users_new_indeces_map[x])

