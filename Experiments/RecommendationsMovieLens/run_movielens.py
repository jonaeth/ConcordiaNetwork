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
    training_data_learn = df_users, df_items, df_ratings_learn_obs

    knowledge_base_factory = KnowledgeBaseFactory(f'data/Transformed/{data_split}/{data_fraction}/learn/')

    print(f'Running Core PSL inference')
    teacher_psl = PSLTeacher(predicates_to_infer=['rating', None],
                             knowledge_base_factory=knowledge_base_factory,
                             **config_concordia)
    teacher_psl.fit(training_data_learn, df_ratings_learn_targets)

    print(f'Running PSL Distribution inference')

    merge_observed_and_predicted_data(f'{self.predicates_folder}/truths/rating_truth.psl',
                                      f'{self.predicates_folder}/observations/rating_obs.psl',
                                      f'{self.predicates_folder}/observations/rating.psl')

    os.makedirs(output_folder, exist_ok=True)
    grounded_rules_path = f'{psl_prediction_folder}/psl_rules_{data_fraction}_{fold_nr}.psl'  # CORE_PSL_OUTPUT_FOLDER
    predicate_database_path = path_to_predicate_file  # PATH_TO_PREDICATE_DATA_PATHS_FILE
    rating_targets_path = f'Data/MovieLens/Transformed/train/{data_fraction}/{fold_nr}/learn/rated_obs.psl'  # f'{psl_prediction_folder}/psl_pred_{data_fraction}_{fold_nr}.psl'
    path_to_save_prob_density_files = f'{output_folder}/prob_denstiy_frac{data_fraction}_{fold_nr}.psl'  # PSL_DISTRIBUTION_OUTPUT_FOLDER
    facts = Facts(predicate_database_path, str(fold_nr), 'learn', data_fraction)
    print('done')

    herbrand_base = HerbrandBase(grounded_rules_path, 'rating')

    with open(rating_targets_path) as fp:
        target_predicate_arguments = [tuple(line.strip().split()[:2]) for line in fp.readlines()]

    print('making inference')
    prob_estimations = get_pdf_estimate_of_targets_integration(herbrand_base, facts, target_predicate_arguments,
                                                               'rating', path_to_save_prob_density_files)
    print('inference is done')
    with open(path_to_save_prob_density_files, 'w') as fp:
       fp.writelines([f"{user_id}\t{item_id}\t{', '.join([str(i) for i in dist])}\n" for (user_id, item_id), dist in
                      zip(target_predicate_arguments, prob_estimations)])

    # predict_psl_distribution(fold_nr=0, data_fraction=data_fraction,
    #                          psl_prediction_folder='core_psl_predictions/movielens',
    #                          output_folder='psl_distribution_predictions/movielens',
    #                          path_to_predicate_file='paths_to_predicate_data_movielens.psl')

#run(5)
#run(10)
#run(20)
#run(50)
#run(80)
run(100)
