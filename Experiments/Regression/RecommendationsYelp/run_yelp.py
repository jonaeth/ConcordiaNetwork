from Concordia.Teacher import PSLTeacher

from data.DataExtraction import DataExtraction
from Experiments.RecommendationsYelp.KnowledgeBaseFactory import KnowledgeBaseFactory
from sklearn.model_selection import train_test_split
from config_concordia import config_concordia



def extract_yelp_data():
    data_exctracted = DataExtraction('Cambridge')
    df_business = data_exctracted.df_business
    df_ratings = data_exctracted.df_ratings
    df_users = data_exctracted.df_users
    return df_business, df_users, df_ratings


def run(data_fraction):
    df_items, df_users, df_ratings = extract_yelp_data()
    df_ratings_learn_obs, df_ratings_learn_targets = train_test_split(df_ratings, test_size=0.1, random_state=0)

    training_data_learn = df_users, df_items, df_ratings_learn_obs

    knowledge_base_factory = KnowledgeBaseFactory('teacher/train/')

    # Teacher
    print(f'Running Core PSL inference')
    teacher_psl = PSLTeacher(predicates_to_infer=['rating', None],
                             knowledge_base_factory=knowledge_base_factory,
                             **config_concordia)
    teacher_psl.fit(training_data_learn, df_ratings_learn_targets)

    print(f'Running PSL Distribution inference')
    predicates_folder = config_concordia['ground_predicates_path']

    predictions = teacher_psl.predict()
    print('inference is done')

# run(5)
# run(10)
# run(20)
# run(50)
# run(80)
run(100)