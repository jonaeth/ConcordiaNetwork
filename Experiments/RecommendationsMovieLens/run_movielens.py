from Concordia.Teacher import PSLTeacher
from DataLoader import DataLoader
from HyperTransformations.DataTransformation import DataTransformation
from Experiments.RecommendationsMovieLens.KnowledgeBaseFactory import KnowledgeBaseFactory
from sklearn.model_selection import train_test_split
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
    # Config
    markov_blanket_file_path = config_concordia['markov_blanket']  # CORE_PSL_OUTPUT_FOLDER

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

    knowledge_base_factory = KnowledgeBaseFactory('teacher/train/')

    print(f'Running Core PSL inference')
    teacher_psl = PSLTeacher(predicates_to_infer=['rating', None],
                             knowledge_base_factory=knowledge_base_factory,
                             **config_concordia)
    teacher_psl.fit(training_data_learn, df_ratings_learn_targets)

    print(f'Running PSL Distribution inference')
    merge_observed_and_predicted_data(f'{self.predicates_folder}/truths/rating.psl',
                                      f'{self.predicates_folder}/observations/rating.psl',
                                      f'{self.predicates_folder}/observations/rating_concatenated.psl')  # TODO Modestas: Fix Problem!

    teacher_psl.predict()
    print('inference is done')
    with open(path_to_save_prob_density_files, 'w') as fp:
       fp.writelines([f"{user_id}\t{item_id}\t{', '.join([str(i) for i in dist])}\n" for (user_id, item_id), dist in
                      zip(target_predicate_arguments, prob_estimations)])

#run(5)
#run(10)
#run(20)
#run(50)
#run(80)
run(100)
