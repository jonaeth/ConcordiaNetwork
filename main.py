from Concordia.Teacher import PSLTeacher
from Concordia.Student import Student
from Experiments.CollectiveActivity.NeuralNetworkModels.BaseNet import BaseNet
from torch.optim import Adam
from Experiments.CollectiveActivity.PredicateBuilder import PredicateBuilder
from Experiments.CollectiveActivity.config_experiment1 import cfg
from Experiments.CollectiveActivity.dataset import return_dataset
from Concordia.ConcordiaNetwork import ConcordiaNetwork
from torch.utils import data
import numpy as np
from Concordia.torch_losses import kl_divergence, cross_entropy
import torch
from Experiments.CollectiveActivity.concordia_config import concordia_config
from Experiments.CollectiveActivity.CollectiveActivityCallback import CollectiveActivityCallback


predicate_file = 'Experiments/CollectiveActivity/data/teacher/model/predicates.psl'
rule_file = 'Experiments/CollectiveActivity/data/teacher/model/model.psl'
train_predicate_folder = 'Experiments/CollectiveActivity/data/teacher/train'
path_to_save_predicates = 'Experiments/CollectiveActivity/data/teacher/train'

knowledge_base_factory = PredicateBuilder(path_to_save_predicates, cfg)

teacher_psl = PSLTeacher(predicates_to_infer=['DOING', None], knowledge_base_factory=knowledge_base_factory, **concordia_config)
teacher_psl.build_model()


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

def actions_accuracy(student_predictions, targets):
    student_predictions_actions, _ = student_predictions
    target_actions, _ = convert_targets_to_right_shape(*targets)
    predicted_actions_labels = torch.argmax(student_predictions_actions, dim=1)
    actions_correct = torch.sum(torch.eq(predicted_actions_labels.int(), target_actions.int()).float())
    actions_accuracy = actions_correct.item() / student_predictions_actions.shape[0]
    return actions_accuracy

def activities_accuracy(student_predictions, targets):
    _, student_predictions_activities = student_predictions
    _, target_activities = convert_targets_to_right_shape(*targets)
    predicted_activities_labels = torch.argmax(student_predictions_activities, dim=1)
    activities_correct = torch.sum(torch.eq(predicted_activities_labels.int(), target_activities.int()).float())
    activities_accuracy = activities_correct.item() / student_predictions_activities.shape[0]
    return activities_accuracy


cfg.backbone='mobilenet'
cfg.backbone='inv3'
base_neural_network = BaseNet(cfg)

params = list(filter(lambda p: p.requires_grad, base_neural_network.parameters()))
optimizer = Adam(params, lr=cfg.train_learning_rate, weight_decay=cfg.weight_decay)
student_nn = Student(base_neural_network, student_target_loss_function, optimizer)

training_set, validation_set=return_dataset(cfg)

training_set_loader = data.DataLoader(training_set)
validation_set_loader = data.DataLoader(validation_set)


callbacks = [CollectiveActivityCallback(cfg.log_path)]
custom_metrics = {'actions_acc': actions_accuracy, 'activities_acc': activities_accuracy}

concordia_network = ConcordiaNetwork(student_nn, teacher_psl, **concordia_config)
concordia_network.fit(training_set_loader, validation_set_loader, callbacks=callbacks, metrics=custom_metrics)
