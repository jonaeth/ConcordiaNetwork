import torch

from Concordia.Teacher import PSLTeacher
from Concordia.Student import Student
from Concordia.ConcordiaNetwork import ConcordiaNetwork
from Experiments.Regression.Utils.metrics_and_loss import RMSE_LOSS, custom_RMSE_LOSS
from torch.optim import Adam
from torch.utils import data
from DataLoader import DataLoader
from Experiments.Regression.RecommendationsMovieLens.KnowledgeBaseFactory import KnowledgeBaseFactory
from sklearn.model_selection import train_test_split
import pandas as pd
from config_concordia import config_concordia
from Experiments.Regression.neural_network_models.ncf import NCF
from config import neural_network_config, optimiser_config, concordia_config, time_str
data_split = 'train'


def extract_movielens_data(data_split, path_to_data):
    data_exctracted = DataLoader(data_split, path_to_data)
    df_movies = data_exctracted.df_movies
    df_ratings = data_exctracted.df_ratings
    df_users = data_exctracted.df_users
    return df_movies, df_users, df_ratings


class MovieLensDataset(data.Dataset):
    def __init__(self, ratings_df, psl_predictions=None):
        self.x, self.y = self._split_rating_data_to_xy(ratings_df)
        self.psl_predictions = torch.tensor([i[1] for i in psl_predictions[0][:len(self.x)]]) if psl_predictions else None #This is rather hacky

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if self.psl_predictions is not None:
            return self.x[index], [self.psl_predictions[index]], self.y[index]
        else:
            return self.x[index], self.y[index]

    def _split_rating_data_to_xy(self, df_ratings):
        x = df_ratings[['user_id', 'item_id']].values
        y = df_ratings[['rating']].values
        return x, y

def run(data_fraction):
    # Load Data
    data_fraction = 100
    path_to_data = "Experiments/Regression/RecommendationsMovieLens/data"

    df_items, df_users, df_ratings_learn = extract_movielens_data('train', path_to_data)
    _, _, df_ratings_validation = extract_movielens_data('valid', path_to_data)
    _, _, df_ratings_eval = extract_movielens_data('test', path_to_data)

    num_users = len(df_users)
    num_items = len(df_items)
    print(f'nr_items {num_items}')
    print(f'nr_users {num_users}')

    print(f'nr_ratings {len(df_ratings_learn) + len(df_ratings_validation) + len(df_ratings_eval)}')

    df_ratings_learn = pd.concat([df_ratings_learn, df_ratings_validation])
    df_ratings_learn = df_ratings_learn.sample(frac=data_fraction / 100, random_state=0)

    test_size = 0.3
    df_ratings_learn_obs, df_ratings_learn_targets = train_test_split(df_ratings_learn,
                                                                      test_size=test_size,
                                                                      random_state=0)
    training_data_learn = df_users, df_items, df_ratings_learn_obs

    # Build predicate files
    knowledge_base_factory = KnowledgeBaseFactory('Experiments/Regression/RecommendationsMovieLens/teacher/train')

    # Teacher
    print(f'Running Core PSL inference')
    teacher_psl = PSLTeacher(predicates_to_infer=['rating'],
                             knowledge_base_factory=knowledge_base_factory,
                             **config_concordia)
    teacher_psl.fit(training_data_learn, df_ratings_learn_targets)

    print(f'Running PSL Distribution inference')

    psl_predictions = teacher_psl.predict()

    with open(f'Experiments/Regression/RecommendationsMovieLens/logs/psl_distribution_predictions_{time_str}.txt', 'w') as fp:
        for pred_arg, dist in zip(psl_predictions[0][0], psl_predictions[0][1]):
            fp.write(f"{pred_arg}_[{','.join(dist.astype(str))}]\n")


    neural_network_config['num_users'] = num_users
    neural_network_config['num_items'] = num_items

    base_neural_network = NCF(**neural_network_config)
    params = list(filter(lambda p: p.requires_grad, base_neural_network.parameters()))
    optimizer = Adam(params, lr=optimiser_config['lr'])

    student_nn = Student(base_neural_network, RMSE_LOSS, optimizer)

    training_data = MovieLensDataset(df_ratings_learn_obs, psl_predictions)
    validation_data = MovieLensDataset(df_ratings_eval)

    training_loader = data.DataLoader(training_data, batch_size=optimiser_config['batch_size'])
    validation_loader = data.DataLoader(validation_data)

    concordia_network = ConcordiaNetwork(student_nn, teacher_psl, **concordia_config)

    concordia_network.fit(training_loader, validation_loader, metrics={'RMSE_LOSS': custom_RMSE_LOSS})
    print('inference is done')
    concordia_network.predict()


def run_neural_network_only(data_fraction):
    # Load Data
    path_to_data = "Experiments/Regression/RecommendationsMovieLens/data"

    df_items, df_users, df_ratings_learn = extract_movielens_data('train', path_to_data)
    _, _, df_ratings_validation = extract_movielens_data('valid', path_to_data)
    _, _, df_ratings_eval = extract_movielens_data('test', path_to_data)

    num_users = len(df_users)
    num_items = len(df_items)
    print(f'nr_items {num_items}')
    print(f'nr_users {num_users}')

    neural_network_config['num_users'] = num_users
    neural_network_config['num_items'] = num_items

    print(f'nr_ratings {len(df_ratings_learn) + len(df_ratings_validation) + len(df_ratings_eval)}')

    df_ratings_learn = pd.concat([df_ratings_learn, df_ratings_validation])
    df_ratings_learn = df_ratings_learn.sample(frac=data_fraction / 100, random_state=0)

    test_size = 0.3
    df_ratings_learn_obs, df_ratings_learn_targets = train_test_split(df_ratings_learn,
                                                                      test_size=test_size,
                                                                      random_state=0)
    training_data_learn = df_users, df_items, df_ratings_learn_obs
    base_neural_network = NCF(**neural_network_config)
    params = list(filter(lambda p: p.requires_grad, base_neural_network.parameters()))
    optimizer = Adam(params, lr=optimiser_config['lr'])

    psl_predictions = []

    with open(f'Experiments/Regression/RecommendationsMovieLens/logs/psl_distribution_predictions_2022_01_23_09_54_19.txt', 'r') as fp:
        for line in fp.readlines():
            atoms, distribution = line.split('_')
            atoms, distribution = eval(atoms), eval(distribution)
            psl_predictions.append((atoms, distribution))

    psl_predictions = [psl_predictions]
    student_nn = Student(base_neural_network, RMSE_LOSS, optimizer)

    training_data = MovieLensDataset(df_ratings_learn_obs, psl_predictions)
    validation_data = MovieLensDataset(df_ratings_eval)

    training_loader = data.DataLoader(training_data, batch_size=optimiser_config['batch_size'])
    validation_loader = data.DataLoader(validation_data)

    concordia_network = ConcordiaNetwork(student_nn, None, **concordia_config)

    concordia_network.fit(training_loader, validation_loader, metrics={'RMSE_LOSS': custom_RMSE_LOSS})
    print('inference is done')
    concordia_network.predict()

#run(5)
#run(10)
#run(20)
#run(50)
#run(80)
run_neural_network_only(100)
