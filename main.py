import os

from torch.utils import data
from torch.optim import Adam
import argparse

from Concordia.Teacher import PSLTeacher
from Concordia.Student import Student
from Concordia.ConcordiaNetwork import ConcordiaNetwork

from Experiments.Classification.CollectiveActivity.config import Config
from Experiments.Classification.CollectiveActivity.NeuralNetworkModels.BaseNet import BaseNet
from Experiments.Classification.CollectiveActivity.KnowledgeBaseFactory import KnowledgeBaseFactory
from Experiments.Classification.CollectiveActivity.collective import return_dataset
from Experiments.Classification.CollectiveActivity.config_concordia import config_concordia
from Experiments.Classification.CollectiveActivity.CollectiveActivityCallback import CollectiveActivityCallback
from Experiments.Classification.CollectiveActivity.CAD_metrics_and_loss import *
from Experiments.Classification.CollectiveActivity.logging import *


def main(backbone, use_gpu, gpu_id):
    # Set up config
    cfg = Config('collective')
    cfg.init_config('Experiments/CollectiveActivity/result/')
    if backbone == 'mobilenet':
        cfg.backbone = 'mobilenet'
    else:
        cfg.backbone = 'inv3'

    cfg.exp_note = 'Collective_train_' + backbone
    cfg.use_gpu = use_gpu
    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(f'{i}' for i in range(gpu_id+1))
        cfg.gpu_id = gpu_id
    show_config(cfg)

    # Teacher
    path_to_save_predicates = 'Experiments/Classification/CollectiveActivity/teacher/train'
    knowledge_base_factory = KnowledgeBaseFactory(path_to_save_predicates, cfg)
    teacher_psl = PSLTeacher(predicates_to_infer=['DOING', None],
                             knowledge_base_factory=knowledge_base_factory,
                             **config_concordia)

    # Student
    base_neural_network = BaseNet(cfg)
    params = list(filter(lambda p: p.requires_grad, base_neural_network.parameters()))
    optimizer = Adam(params, lr=cfg.train_learning_rate, weight_decay=cfg.weight_decay)
    student_nn = Student(base_neural_network, student_target_loss_function, optimizer)

    # Data
    training_set, validation_set = return_dataset(cfg)

    # Setting metrics to be evaluated during Training and Testing
    callbacks = [CollectiveActivityCallback(cfg.log_path)]
    custom_metrics = {'actions_acc': actions_accuracy, 'activities_acc': activities_accuracy}

    concordia_network = ConcordiaNetwork(student_nn, teacher_psl, **config_concordia)


    concordia_network.fit(data.DataLoader(training_set),
                          data.DataLoader(validation_set),
                          callbacks=callbacks,
                          metrics=custom_metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collective Activity Detection using Concordia')
    parser.add_argument('--backbone',
                        default='inv3',
                        help='Backbone of the base model: Choose mobilenet or inception')
    parser.add_argument('--use_gpu',
                        default=False,
                        type=bool,
                        help='DNN can be trained on gpu, which is significantly faster. Default is False.')
    parser.add_argument('--gpu_id',
                        default=0,
                        type=int,
                        help='Specify which gpu to use. Default is 0.')
    args = parser.parse_args()
    main(backbone=args.backbone, use_gpu=args.use_gpu, gpu_id=args.gpu_id)
