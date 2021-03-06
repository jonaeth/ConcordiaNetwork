from Concordia.Teacher import PSLTeacher
from Concordia.Student import Student
from Concordia.ConcordiaNetwork import ConcordiaNetwork
from Experiments.Regression.Utils.metrics_and_loss import RMSE_LOSS, custom_RMSE_LOSS

from torch.utils import data
from data.DataExtraction import DataExtraction
from Experiments.Regression.RecommendationsYelp.KnowledgeBaseFactory import KnowledgeBaseFactory
from sklearn.model_selection import train_test_split
from config_concordia import config_concordia
from config import neural_network_config, optimiser_config, concordia_config
from torch.optim import Adam
from torch.utils import data
from Experiments.Regression.neural_network_models.ncf import NCF


def extract_yelp_data(path_to_data):
    data_exctracted = DataExtraction(path_to_data, 'Cambridge')
    df_business = data_exctracted.df_business
    df_ratings = data_exctracted.df_ratings
    df_users = data_exctracted.df_users
    return df_business, df_users, df_ratings


class YelpDataset(data.Dataset):
    def __init__(self, ratings_df, psl_predictions=None):
        self.x, self.y = self._split_rating_data_to_xy(ratings_df)
        self.psl_predictions = psl_predictions[0][1][:len(self.x)] if psl_predictions else None  # This is rather hacky

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if self.psl_predictions:
            return self.x[index], [self.psl_predictions[index]], self.y[index]
        else:
            return self.x[index], self.y[index]

    def _split_rating_data_to_xy(self, df_ratings):
        x = df_ratings[['user_id', 'item_id']].values
        y = df_ratings[['rating']].values
        return x, y


def run(data_fraction):
    # Load Data
    data_fraction = 5
    path_to_data = "Experiments/Regression/RecommendationsYelp/data"

    df_items, df_users, df_ratings = extract_yelp_data(path_to_data)

    num_users = len(df_users)
    num_items = len(df_items)
    print(f'nr_items {num_items}')
    print(f'nr_users {num_users}')

    print(f'nr_ratings {len(df_ratings)}')

    eval_size = 0.1

    df_ratings_train, df_ratings_eval = train_test_split(df_ratings, test_size=eval_size, random_state=0)

    test_size = 0.3

    df_ratings_learn_obs, df_ratings_learn_targets = train_test_split(df_ratings_train, test_size=test_size, random_state=0)
    df_ratings_learn_obs = df_ratings_learn_obs.sample(frac=data_fraction / 100, random_state=0)

    training_data_learn = df_users, df_items, df_ratings_learn_obs

    knowledge_base_factory = KnowledgeBaseFactory('Experiments/Regression/RecommendationsYelp/teacher/train')

    # Teacher
    print(f'Running Core PSL inference')
    teacher_psl = PSLTeacher(predicates_to_infer=['rating'],
                             knowledge_base_factory=knowledge_base_factory,
                             **config_concordia)
    teacher_psl.fit(training_data_learn, df_ratings_learn_targets)

    print(f'Running PSL Distribution inference')

    psl_predictions = teacher_psl.predict()
    print('inference is done')

    neural_network_config['num_users'] = num_users
    neural_network_config['num_items'] = num_items

    base_neural_network = NCF(**neural_network_config)
    params = list(filter(lambda p: p.requires_grad, base_neural_network.parameters()))
    optimizer = Adam(params, lr=optimiser_config['lr'])

    student_nn = Student(base_neural_network, RMSE_LOSS, optimizer)

    training_data = YelpDataset(df_ratings_learn_obs, psl_predictions)
    validation_data = YelpDataset(df_ratings_eval)

    training_loader = data.DataLoader(training_data, batch_size=optimiser_config['batch_size'])
    validation_loader = data.DataLoader(validation_data)

    concordia_network = ConcordiaNetwork(student_nn, teacher_psl, **concordia_config)

    concordia_network.fit(training_loader, validation_loader, metrics={'RMSE_LOSS': custom_RMSE_LOSS})
    print('inference is done')
    concordia_network.predict()

# run(5)
# run(10)
# run(20)
# run(50)
# run(80)
run(100)
