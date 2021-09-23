import os
import numpy as np
import pandas as pd
from pslpython.model import Model
from pslpython.partition import Partition
from pslpython.predicate import Predicate
from pslpython.rule import Rule
from dataset import return_dataset

DATA_DIR = os.path.join('..', 'data', MODEL_NAME)

ADDITIONAL_PSL_OPTIONS = {
    'log4j.threshold': 'OFF',
    'votedperceptron.numsteps': '2'
}

NUM_ACTIONS = 8


def run_psl(model=None):
    data_path_to_obs = 'predictions_for_psl_stage1/predicates'
    if not model:
        model = build_psl()

    model.get_predicate('Local').clear_data().add_data_file(Partition.OBSERVATIONS,
                                                            f'{data_path_to_obs}/local_act_obs.txt')
    model.get_predicate('GlobalAct').clear_data().add_data_file(Partition.OBSERVATIONS,
                                                                f'{data_path_to_obs}/global_act_obs.txt')
    model.get_predicate('Close').clear_data().add_data_file(Partition.OBSERVATIONS,
                                                            f'{data_path_to_obs}/close_obs.txt')
    #model.get_predicate('Seq').clear_data().add_data_file(Partition.OBSERVATIONS,
    #                                                      f'{data_path_to_obs}/Sequence.txt')
    model.get_predicate('Same').clear_data().add_data_file(Partition.OBSERVATIONS,
                                                           f'{data_path_to_obs}/same_obs.txt')
    model.get_predicate('Doing').clear_data().add_data_file(Partition.TARGETS,
                                                            f'{data_path_to_obs}/doing_targets.txt')

    model.get_predicate('Doing').add_data_file(Partition.TRUTH, f'{data_path_to_obs}/doing_truth.txt')

    model.learn(additional_cli_optons=ADDITIONAL_CLI_OPTIONS, psl_config=ADDITIONAL_PSL_OPTIONS)

    results = model.infer(additional_cli_optons=ADDITIONAL_CLI_OPTIONS, psl_config=ADDITIONAL_PSL_OPTIONS)
    return model, results[model.get_predicate('Doing')].sort_values(by=[0, 1])



def get_psl_predictions(psl_model, bboxes, action_predictions, ground_truths):
    from psl.predicate_generator import build_all_predicates
    df = build_all_predicates(bboxes, action_predictions, ground_truths)
    model, predictions = run_psl(psl_model)
    return model, np.array([x for x in predictions[[0, 'truth']].groupby(0).apply(lambda x: np.array(x['truth'])).to_numpy()])







