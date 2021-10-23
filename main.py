from Concordia.Teacher import PSLTeacher
from Experiments.CollectiveActivity.CollectiveActivityNet import CollectiveActivityNet
from Experiments.CollectiveActivity.NeuralNetworkModels.BaseNet import BaseNet
from torch.optim import Adam
from Experiments.CollectiveActivity.PredicateBuilder import PredicateBuilder
from Experiments.CollectiveActivity.config_experiment1 import cfg
from Experiments.CollectiveActivity.dataset import return_dataset
from Concordia.ConcordiaNetwork import ConcordiaNetwork
from torch.utils import data
import numpy as np
from Experiments.torch_losses import kl_divergence, cross_entropy
import torch
from Experiments.CollectiveActivity.concordia_config import concordia_config


predicate_file = 'Experiments/CollectiveActivity/data/teacher/model/predicates.psl'
rule_file = 'Experiments/CollectiveActivity/data/teacher/model/model.psl'
train_predicate_folder = 'Experiments/CollectiveActivity/data/teacher/train'

path_to_save_predicates = 'Experiments/CollectiveActivity/data/teacher/train'
teacher_psl = PSLTeacher(predicate_to_infer='DOING')
teacher_psl.build_model(predicate_file=predicate_file, rules_file=rule_file)


def convert_targets_to_right_shape(targets_actions, targets_activities):
    batch_size = targets_actions.shape[0]
    max_number_bboxes = targets_actions.shape[2]
    num_frames = targets_actions.shape[1]

    targets_actions = targets_actions.reshape((batch_size, num_frames, max_number_bboxes))
    targets_activities = targets_activities.reshape((batch_size, num_frames))

    targets_actions_nopad_same_shape = []
    targets_actions_nopad = []
    for b in range(batch_size):
        actions_of_batch = []
        number_of_bboxes_per_frame = [len([action for action in frame if action != -1]) for frame in targets_actions[b].tolist()]
        for i, N in enumerate(number_of_bboxes_per_frame):
            targets_actions_nopad.append(targets_actions[b][i, :N])
            actions_of_batch.append(targets_actions[b][i, :N])
        targets_actions_nopad_same_shape.append(actions_of_batch)

    targets_actions = torch.cat(targets_actions_nopad, dim=0).reshape(-1, )  # ALL_N,
    targets_activities = targets_activities.reshape(-1, )

    return targets_actions, targets_activities


def teacher_student_loss_function(student_predictions, teacher_predictions):
    student_predictions_actions, student_predictions_activities = student_predictions
    teacher_predictions = np.array([x for x in teacher_predictions[[0, 'truth']].groupby(0).apply(lambda x: np.array(x['truth'])).to_numpy()]) #TODO figure out how to remove the 0 if possible
    teacher_predictions = torch.Tensor(teacher_predictions)
    kl_divergnece_loss = kl_divergence(student_predictions_actions, teacher_predictions)
    return kl_divergnece_loss

def student_target_loss_function(student_predictions, targets):
    student_predictions_actions, student_predictions_activities = student_predictions
    target_actions, target_activities = targets
    target_actions, target_activities = convert_targets_to_right_shape(target_actions, target_activities)
    return cross_entropy(student_predictions_actions, target_actions) + cross_entropy(student_predictions_activities, target_activities)

base_neural_network = BaseNet(cfg)
optimizer = Adam(base_neural_network.parameters(), lr=0.001)
student_nn = CollectiveActivityNet(base_neural_network, optimizer)

predicate_builder = PredicateBuilder(path_to_save_predicates, cfg)
training_set, validation_set=return_dataset(cfg)

training_set_loader = data.DataLoader(training_set)

concordia_network = ConcordiaNetwork(student_nn, teacher_psl, predicate_builder, teacher_student_loss_function, student_target_loss_function, False)

concordia_network.fit(training_set_loader)
