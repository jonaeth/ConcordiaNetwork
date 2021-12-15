from Concordia.Teacher import PSLTeacher
from DataLoader import DataLoader
from Experiments.RecommendationsMovieLens.KnowledgeBaseFactory import KnowledgeBaseFactory
from sklearn.model_selection import train_test_split
import pandas as pd
from config_concordia import config_concordia
data_split = 'train'


def extract_movielens_data(data_split, path_to_data):
    data_exctracted = DataLoader(data_split, path_to_data)
    df_movies = data_exctracted.df_movies
    df_ratings = data_exctracted.df_ratings
    df_users = data_exctracted.df_users
    return df_movies, df_users, df_ratings


def merge_observed_and_predicted_data(psl_pred_path, rating_obs_path, output):
    df_pred = pd.read_csv(psl_pred_path, sep='\t', header=None)
    df_ratings_obs = pd.read_csv(rating_obs_path, sep='\t', header=None)
    df = pd.concat([df_pred, df_ratings_obs]).drop_duplicates()
    df.to_csv(output, sep='\t', header=None, index=False)


def run(data_fraction):
    # Load Data
    path_to_data = "Experiments/RecommendationsMovieLens/data"
    df_items, df_users, df_ratings_learn = extract_movielens_data('train', path_to_data)
    _, _, df_ratings_validation = extract_movielens_data('valid', path_to_data)
    _, _, df_ratings_eval = extract_movielens_data('test', path_to_data)

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

    # Build predicate files
    knowledge_base_factory = KnowledgeBaseFactory('Experiments/RecommendationsMovieLens/teacher/train')

    # Teacher
    print(f'Running Core PSL inference')
    teacher_psl = PSLTeacher(predicates_to_infer=['rating'],
                             knowledge_base_factory=knowledge_base_factory,
                             **config_concordia)
    teacher_psl.fit(training_data_learn, df_ratings_learn_targets)

    print(f'Running PSL Distribution inference')
    predicates_folder = config_concordia['ground_predicates_path']

    predictions = teacher_psl.predict()
    print('inference is done')

#run(5)
#run(10)
#run(20)
#run(50)
#run(80)
run(100)
