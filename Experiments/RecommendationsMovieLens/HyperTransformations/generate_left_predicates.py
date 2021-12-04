import pandas as pd
import turicreate as tc
import os
import smurff
from scipy.sparse import csr_matrix
import numpy as np


def build_pearson_sim_predicates(train_data, path_to_save):
    item_based_model = tc.recommender.item_similarity_recommender.create(train_data, user_id='user_id',
                                                                        item_id='item_id', target='rating', similarity_type='pearson')

    user_based_model = tc.recommender.item_similarity_recommender.create(train_data, user_id='item_id',
                                                                        item_id='user_id', target='rating', similarity_type='pearson')

    sim_users_pearson = user_based_model.get_similar_items(k=50)
    sim_items_pearson = item_based_model.get_similar_items(k=50)

    sim_users_pearson.to_dataframe()[['user_id', 'similar']].to_csv(f'{path_to_save}/sim_pearson_users_obs.psl', header=False, index=False, sep='\t')
    sim_items_pearson.to_dataframe()[['item_id', 'similar']].to_csv(f'{path_to_save}/sim_pearson_items_obs.psl', header=False, index=False, sep='\t')


def build_sim_cosine_items(train_data, path_to_save):
    item_based_model = tc.recommender.item_similarity_recommender.create(train_data, user_id='user_id',
                                                                         item_id='item_id', target='rating',
                                                                         similarity_type='cosine')
    sim_items_cosine = item_based_model.get_similar_items(k=50)
    sim_items_cosine.to_dataframe()[['item_id', 'similar']].to_csv(f'{path_to_save}/sim_cosine_items_obs.psl', header=False, index=False, sep='\t')


def build_sim_cosine_users(train_data, path_to_save):
    user_based_model = tc.recommender.item_similarity_recommender.create(train_data, user_id='item_id',
                                                                         item_id='user_id', target='rating',
                                                                         similarity_type='cosine')
    sim_users_cosine = user_based_model.get_similar_items(k=50)
    sim_users_cosine.to_dataframe()[['user_id', 'similar']].to_csv(f'{path_to_save}/sim_cosine_users_obs.psl',
                                                                    header=False, index=False, sep='\t')


def build_pearson_items_predictions(train_data, test_data, path_to_save):
    item_based_model = tc.recommender.item_similarity_recommender.create(train_data, user_id='user_id',
                                                                        item_id='item_id', target='rating', similarity_type='pearson')
    predictions_test = item_based_model.predict(test_data)
    predictions_train = item_based_model.predict(train_data)
    test_data_df = test_data.to_dataframe()
    train_data_df = train_data.to_dataframe()
    test_data_df['pearson_predictions'] = predictions_test
    train_data_df['pearson_predictions'] = predictions_train
    test_data_df['pearson_predictions'] = test_data_df['pearson_predictions'].clip(0, 5)
    train_data_df['pearson_predictions'] = train_data_df['pearson_predictions'].clip(0, 5)
    df = pd.concat([train_data_df, test_data_df])
    df[['user_id', 'item_id', 'pearson_predictions']].to_csv(f'{path_to_save}/item_pearson_rating_obs.psl', header=None, index=False, sep='\t')


def build_sim_mf_cosine_users(train_data, path_to_save):
    user_mf_cosine_model = tc.recommender.factorization_recommender.create(train_data, user_id='item_id',
                                                                         item_id='user_id', target='rating',
                                                                         num_factors=8, max_iterations=10)
    sim_users_cosine = user_mf_cosine_model.get_similar_items(k=50)
    sim_users_cosine.to_dataframe()[['user_id', 'similar']].to_csv(f'{path_to_save}/sim_mf_cosine_users_obs.psl',
                                                                    header=False, index=False, sep='\t')


def build_sim_mf_cosine_items(train_data, path_to_save):
    user_mf_cosine_model = tc.recommender.factorization_recommender.create(train_data, user_id='user_id',
                                                                         item_id='item_id', target='rating',
                                                                         num_factors=8, max_iterations=10)
    sim_users_cosine = user_mf_cosine_model.get_similar_items(k=50)
    sim_users_cosine.to_dataframe()[['item_id', 'similar']].to_csv(f'{path_to_save}/sim_mf_cosine_items_obs.psl',
                                                                   header=False, index=False, sep='\t')


def build_mf_sgd_predictions(train_data, test_data, path_to_save):
    sgd_mf_model = tc.recommender.factorization_recommender.create(train_data, user_id='user_id',
                                                                   item_id='item_id', target='rating', 
                                                                   num_factors=8, max_iterations=10)

    predictions_test = sgd_mf_model.predict(test_data)
    predictions_train = sgd_mf_model.predict(train_data)
    test_data_df = test_data.to_dataframe()
    train_data_df = train_data.to_dataframe()
    test_data_df['sgd_predictions'] = predictions_test
    train_data_df['sgd_predictions'] = predictions_train
    test_data_df['sgd_predictions'] = test_data_df['sgd_predictions'].clip(0, 5)
    train_data_df['sgd_predictions'] = train_data_df['sgd_predictions'].clip(0, 5)
    df = pd.concat([train_data_df, test_data_df])
    df[['user_id', 'item_id', 'sgd_predictions']].to_csv(f'{path_to_save}/sgd_rating_obs.psl',
                                                         header=None,
                                                         index=False, sep='\t')


def create_an_empty_file(path_to_data):
    with open(f"{path_to_data}/empty_obs.psl", 'w'):
        pass


def compute_predicates(path_to_data):

    path_to_train = f'{path_to_data}/rating_obs.psl'
    path_to_test = f'{path_to_data}/rating_truth.psl'
    train_data_learn_df = pd.read_csv(path_to_train, header=None, names=['user_id', 'item_id', 'rating'], sep='\t')
    test_data_learn_df = pd.read_csv(path_to_test, header=None, names=['user_id', 'item_id', 'rating'], sep='\t')

    train_data = tc.SFrame(train_data_learn_df)
    test_data = tc.SFrame(test_data_learn_df)

    rate_with_bpmf(train_data_learn_df, test_data_learn_df, path_to_data)
    build_pearson_sim_predicates(train_data, path_to_data)
    build_pearson_items_predictions(train_data, test_data, path_to_data)
    build_mf_sgd_predictions(train_data, test_data, path_to_data)
    create_an_empty_file(path_to_data)
    build_sim_cosine_items(train_data, path_to_data)
    build_sim_cosine_users(train_data, path_to_data)
    build_sim_mf_cosine_users(train_data, path_to_data)
    build_sim_mf_cosine_items(train_data, path_to_data)


def rate_with_bpmf(train_data_learn_df, test_data_learn_df, path_to_data):
    df_user = pd.read_csv(f"{path_to_data}/user_obs.psl", header=None, names=['user_id'], sep='\t')
    df_item = pd.read_csv(f"{path_to_data}/item_obs.psl", header=None, names=['user_id'], sep='\t')
    nr_of_users = len(df_user)
    nr_of_items = len(df_item)
    rating_train_matrix = __build_csr_matrix_from_ratings(train_data_learn_df, nr_of_users, nr_of_items)
    rating_test_matrix = __build_csr_matrix_from_ratings(test_data_learn_df, nr_of_users, nr_of_items)
    trainSession = smurff.BPMFSession(
        Ytrain=rating_train_matrix,
        Ytest=rating_train_matrix+rating_test_matrix,
        num_latent=32,
        burnin=20,
        nsamples=80,
        verbose=0)
    predictions = trainSession.run()
    y_predictions = [(prediction.coords[0], prediction.coords[1], prediction.pred_avg) for prediction in predictions]
    pd.DataFrame(y_predictions).to_csv(f"{path_to_data}/bpmf_rating_obs.psl", sep='\t', header=False, index=False)


def __build_csr_matrix_from_ratings(df_ratings, n_users, n_items):
    y_train_matrix = np.zeros((n_users, n_items))
    for user_id, item_id, rating in df_ratings.values:
        y_train_matrix[int(user_id), int(item_id)] = rating
    return csr_matrix(y_train_matrix)
    

