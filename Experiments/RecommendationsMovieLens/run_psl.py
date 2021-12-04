import os

from pslpython.model import Model
from pslpython.partition import Partition
from pslpython.predicate import Predicate
from pslpython.rule import Rule
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np


ADDITIONAL_PSL_OPTIONS = {
    'log4j.threshold': 'INFO'
}

ADDITIONAL_CLI_OPTIONS = [
    # '--postgres',
    "--groundrules "
]

def merge_observed_and_predicted_data(psl_pred_path, rating_obs_path, output):
    df_pred = pd.read_csv(psl_pred_path, sep='\t', header=None)
    df_ratings_obs = pd.read_csv(rating_obs_path, sep='\t', header=None)
    df = pd.concat([df_pred, df_ratings_obs]).drop_duplicates()
    df.to_csv(output, sep='\t', header=None, index=False)


def run(fold, frac_data, rule_list, predicate_list, path_to_data, output_path):
    path_to_learn = f'{path_to_data}/{frac_data}/{fold}/learn'#
    path_to_eval = f'{path_to_data}/{frac_data}/{fold}/eval'
    os.makedirs(output_path, exist_ok=True)
    output_ground_rules_path = f'{output_path}/psl_rules_{frac_data}_{fold}.psl'#
    output_infered_ratings_path = f'{output_path}/psl_pred_{frac_data}_{fold}.psl'
    psl_model = Model(f'psl-{fold}-{frac_data}')
    print('model built')
    learn_model(psl_model, rule_list, predicate_list, path_to_learn, output_path)
    print('model learned')
    #predict_eval_data(psl_model, path_to_eval, output_infered_ratings_path)
    merge_observed_and_predicted_data(f'{path_to_learn}/rating_truth.psl', f'{path_to_learn}/rating_obs.psl', f'{path_to_learn}/rating_truth_concated.psl')
    print('model evaluation completed')
    get_all_grounded_rules(psl_model, path_to_learn, output_ground_rules_path)
    print('model rules grounded')


def load_predicates(psl_model, predicate_list):
    for predicate in predicate_list:
        predicate_name = predicate.split('/')[0]
        predicate_size = int(predicate.split('/')[1][0])
        predicate_closed = True if predicate.split(': ')[1] == 'closed' else False
        predicate_obj = Predicate(predicate_name, closed=predicate_closed, size=predicate_size)
        psl_model.add_predicate(predicate_obj)


def load_observed_data(psl_model, data_path):
    for predicate_name in psl_model.get_predicates().keys():
        psl_model.get_predicate(predicate_name).clear_data().add_data_file(Partition.OBSERVATIONS, f'{data_path}/{predicate_name.lower()}_obs.psl')


def load_rules(psl_model, rule_list):
    for rule in rule_list:
        psl_model.add_rule(Rule(rule))


def learn_model(psl_model, rule_list, predicate_list, path_to_learn, folder_to_save_learned_weights):
    load_predicates(psl_model, predicate_list)
    load_rules(psl_model, rule_list)
    load_observed_data(psl_model, path_to_learn)
    psl_model.get_predicate('rating').add_data_file(Partition.TARGETS, f'{path_to_learn}/rating_targets.psl')
    psl_model.get_predicate('rating').add_data_file(Partition.TRUTH, f'{path_to_learn}/rating_truth.psl')
    psl_model.learn(jvm_options=['-Xms4096M', '-Xmx12000M'])
    psl_model._write_rules(folder_to_save_learned_weights)
    return psl_model

def learn_model_kfold(psl_model, rule_list, predicate_list, path_to_learn, folder_to_save_learned_weights):
    load_predicates(psl_model, predicate_list)
    load_rules(psl_model, rule_list)
    load_observed_data(psl_model, path_to_learn)
    df_obs = pd.read_csv(f'{path_to_learn}/rating_obs.psl', sep='\t', header=None, names=['user_id', 'item_id', 'ratings'], dtype={'user_id': int, 'item_id': int})
    df_targets = pd.read_csv(f'{path_to_learn}/rating_truth.psl', sep='\t', header=None, names=['user_id', 'item_id', 'ratings'], dtype={'user_id': int, 'item_id': int})
    df = pd.concat([df_obs, df_targets]).reset_index(drop=True)
    kf = KFold(n_splits=1, shuffle=False)
    learned_rules = []
    for train_index, test_index in kf.split(df):
        df_train = df.iloc[train_index].reset_index(drop=True)
        df_targets = df.iloc[test_index].reset_index(drop=True)
        psl_model.get_predicate('rating').clear_data()
        psl_model.get_predicate('rating').add_data(Partition.OBSERVATIONS, [[int(i[0]), int(i[1]), i[2]] for i in df_train.values])
        psl_model.get_predicate('rating').add_data(Partition.TARGETS, [[int(i[0]), int(i[1])] for i in df_targets[['user_id', 'item_id']].values])
        psl_model.get_predicate('rating').add_data(Partition.TRUTH, [[int(i[0]), int(i[1]), i[2]] for i in df_targets.values])
        psl_model.learn(jvm_options=['-Xms4096M', '-Xmx12000M'])
        learned_rules.append([rule.weight() for rule in psl_model.get_rules()])

    learned_rules = np.array(learned_rules).mean(axis=0)
    for weight, rule in zip(learned_rules, psl_model.get_rules()):
        rule.set_weight(weight)

    psl_model._write_rules(folder_to_save_learned_weights)

def predict_eval_data(psl_model, path_to_eval, path_to_save_results):
    load_observed_data(psl_model, path_to_eval)
    psl_model.get_predicate('rating').add_data_file(Partition.TARGETS, f'{path_to_eval}/rating_targets.psl')
    results = psl_model.infer(jvm_options=['-Xms4096M', '-Xmx12000M'])
    results[psl_model.get_predicate('rating')].to_csv(path_to_save_results, sep='\t', header=False, index=False)


def get_all_grounded_rules(psl_model, path_to_eval, output_path):
    #df = pd.read_csv(f'{path_to_eval}/rating_obs.psl', sep='\t', names=['user_id', 'item_id', 'rating'])
    #df[['user_id', 'item_id']].to_csv(f'{path_to_eval}/rated_learn_obs.psl', sep='\t', header=None, index=None)
    load_observed_data(psl_model, path_to_eval)
    psl_model.get_predicate('rating').clear_data()
    psl_model.get_predicate('rating').add_data_file(Partition.TARGETS, f'{path_to_eval}/rated_obs.psl') #All ratings as targets
    psl_model.infer(additional_cli_optons=[f'--groundrules', output_path], jvm_options=['-Xms4096M', '-Xmx12000M'])


#run(0, 5)