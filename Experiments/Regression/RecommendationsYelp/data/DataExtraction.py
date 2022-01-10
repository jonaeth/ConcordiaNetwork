import pandas as pd
import json


class DataExtraction:
    def __init__(self, city_of_interest, fraction_to_sample=1.0):
        self.city_of_interest = city_of_interest
        self.df_business = self.__extract_business_data('Data/Yelp/Raw/yelp_academic_dataset_business.json')
        self.df_ratings = self.__extract_business_ratings('Data/Yelp/Raw/yelp_academic_dataset_review-003.json')
        self.__resample_ratings(fraction_to_sample)
        self.df_users = self.__extract_users_data('Data/Yelp/Raw/yelp_academic_dataset_user-002.json')
        self.__replace_hash_ids_with_incremental_ids()

    def __resample_ratings(self, fraction_to_sample):
        self.df_ratings = self.df_ratings.sample(frac=fraction_to_sample, random_state=0)
        business_ids_to_keep = set(self.df_business.item_id)
        self.df_business = self.df_business[self.df_business['item_id'].isin(business_ids_to_keep)]

    def __extract_business_data(self, path):
        print(f'Reading business data {path}')
        with open(path, 'r', encoding='utf-8') as fp:
            business_information_raw = fp.readlines()
        business_information = [json.loads(line) for line in business_information_raw]
        del business_information_raw
        business_information_extracted = [line for line in business_information if line['city'] == self.city_of_interest]
        df_business = pd.DataFrame(
            [{'item_id': business['business_id'], 'categories': business['categories'].split('&')} for business in
             business_information_extracted])
        return df_business

    def __extract_business_ratings(self, path):
        print(f'Reading business rating data {path}')
        with open(path, 'r', encoding='utf-8') as fp:
            review_information_raw = fp.readlines()
        review_information = []
        business_ids_of_interest = set(self.df_business['item_id'].values)
        for review in review_information_raw:
            review = json.loads(review)
            if review['business_id'] in business_ids_of_interest:
                review_information.append(
                    {'item_id': review['business_id'], 'user_id': review['user_id'], 'rating': review['stars']}
                )
        df_ratings = pd.DataFrame(review_information).drop_duplicates(subset=['item_id', 'user_id'], keep='last')
        return df_ratings

    def __extract_users_data(self, path):
        print(f'Reading user data {path}')
        with open(path, 'r', encoding='utf-8') as fp:
            user_information_raw = fp.readlines()
        user_information = [json.loads(user_row) for user_row in user_information_raw]
        del user_information_raw
        users_of_interest = set(self.df_ratings['user_id'].values)
        user_df = pd.DataFrame([{'user_id': user_row['user_id'], 'friends': user_row['friends'].strip().split(', ')} for user_row in user_information if user_row['user_id'] in users_of_interest])
        return user_df

    def __replace_hash_ids_with_incremental_ids(self):
        businessid_idx_map = {business_id: i for i, business_id in
                              enumerate(self.df_ratings['item_id'].drop_duplicates().values)}
        userid_idx_map = {user_id: i for i, user_id in enumerate(self.df_ratings['user_id'].drop_duplicates().values)}
        users_of_interest = set(userid_idx_map.keys())
        self.df_ratings['item_id'] = self.df_ratings.item_id.apply(lambda x: businessid_idx_map[x])
        self.df_ratings['user_id'] = self.df_ratings.user_id.apply(lambda x: userid_idx_map[x])
        self.df_business = self.df_business[self.df_business['item_id'].isin(set(businessid_idx_map.keys()))]
        self.df_users = self.df_users[self.df_users['user_id'].isin(set(userid_idx_map.keys()))]

        self.df_business['item_id'] = self.df_business['item_id'].apply(lambda x: businessid_idx_map[x])
        self.df_users['user_id'] = self.df_users['user_id'].apply(lambda x: userid_idx_map[x])
        self.df_users['friends_idx'] = self.df_users['friends'].apply(lambda x: [userid_idx_map[friend] for friend in x if friend in users_of_interest])

