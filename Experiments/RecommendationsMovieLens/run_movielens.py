from Concordia.Teacher import PSLTeacher
from DataLoader import DataLoader
from HyperTransformations.DataTransformation import DataTransformation
from Experiments.RecommendationsMovieLens.KnowledgeBaseFactory import KnowledgeBaseFactory
from sklearn.model_selection import train_test_split
from compute_psl_distribution_yelp import predict_psl_distribution
from run_psl import run as run_core_psl
import pandas as pd
from config_concordia import config_concordia

data_split = 'train'

def extract_movielens_data(data_split, path_to_data):
    data_exctracted = DataLoader(data_split, path_to_data)
    df_movies = data_exctracted.df_movies
    df_ratings = data_exctracted.df_ratings
    df_users = data_exctracted.df_users
    return df_movies, df_users, df_ratings


def run(data_fraction):

    df_items, df_users, df_ratings_learn = extract_movielens_data('train', 'data/ml-100k')
    _, _, df_ratings_validation = extract_movielens_data('valid', 'data/ml-100k')
    _, _, df_ratings_eval = extract_movielens_data('test', 'data/ml-100k')

    print(f'nr_items {len(df_items)}')
    print(f'nr_users {len(df_users)}')
    print(f'nr_ratings {len(df_ratings_learn) + len(df_ratings_validation) + len(df_ratings_eval)}')

    df_ratings_learn = pd.concat([df_ratings_learn, df_ratings_validation])
    df_ratings_learn = df_ratings_learn.sample(frac=data_fraction / 100, random_state=0)

    test_size = 0.3
    df_ratings_learn_obs, df_ratings_learn_targets = train_test_split(df_ratings_learn,
                                                                      test_size=test_size,
                                                                      random_state=0)

    knowledge_base_factory_learn = KnowledgeBaseFactory(f'data/Transformed/{data_split}/{data_fraction}/learn/')
    training_data_learn = df_users, df_items, df_ratings_learn_obs
    knowledge_base_factory_learn.build_predicates(training_data_learn,
                                                  df_ratings_learn_targets)

    print(f'Running Core PSL inference')
    # teacher_psl = PSLTeacher(predicates_to_infer=['rating', None],
    #                          knowledge_base_factory=knowledge_base_factory,
    #                          **config_concordia)

    # path_to_learn = f'{path_to_data}/{frac_data}/{fold}/learn'
    '''
    os.makedirs(output_path, exist_ok=True)
    output_ground_rules_path = f'{output_path}/psl_rules_{frac_data}_{fold}.psl'
    learn_model(psl_model, rule_list, predicate_list, path_to_learn, output_path)
    merge_observed_and_predicted_data(f'{path_to_learn}/rating_truth.psl', f'{path_to_learn}/rating_obs.psl', f'{path_to_learn}/rating_truth_concated.psl')
    get_all_grounded_rules(psl_model, path_to_learn, output_ground_rules_path)
    '''
    run_core_psl(fold=0, frac_data=data_fraction, output_path='core_psl_predictions/movielens',
                 predicate_list=predicate_list, rule_list=rule_list, path_to_data='Data/MovieLens/Transformed/train')

    print(f'Running PSL Distribution inference')

    merge_observed_and_predicted_data(f'{self.predicates_folder}/truths/rating_truth.psl',
                                      f'{self.predicates_folder}/observations/rating_obs.psl',
                                      f'{self.predicates_folder}/observations/rating.psl')

    predict_psl_distribution(fold_nr=0, data_fraction=data_fraction,
                             psl_prediction_folder='core_psl_predictions/movielens',
                             output_folder='psl_distribution_predictions/movielens',
                             path_to_predicate_file='paths_to_predicate_data_movielens.psl')

#run(5)
#run(10)
#run(20)
#run(50)
#run(80)
run(100)
