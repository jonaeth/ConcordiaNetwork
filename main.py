from Teacher import PSLTeacher
from Experiments.CollectiveActivity.CollectiveActivityNet import CollectiveActivityNet
from Experiments.CollectiveActivity.NeuralNetworkModels.BaseNet import BaseNet
from torch.optim import Adam
from Experiments.CollectiveActivity.PredicateBuilder import PredicateBuilder
from Experiments.CollectiveActivity.config_experiment1 import cfg
from Experiments.CollectiveActivity.dataset import return_dataset
from ConcordiaNetwork import ConcordiaNetwork

predicate_file = 'Experiments/CollectiveActivity/data/teacher/model/predicates.psl'
rule_file = 'Experiments/CollectiveActivity/data/teacher/model/model.psl'
train_predicate_folder = 'Experiments/CollectiveActivity/data/teacher/train'

path_to_save_predicates = 'Experiments/CollectiveActivity/data/teacher/train'
teacher_psl = PSLTeacher()
teacher_psl.build_model(predicate_file=predicate_file, rules_file=rule_file)

base_neural_network = BaseNet(cfg)
optimizer = Adam(base_neural_network.parameters(), lr=0.001)
student_nn = CollectiveActivityNet(base_neural_network, optimizer)

predicate_builder = PredicateBuilder(path_to_save_predicates, cfg)
training_set, validation_set=return_dataset(cfg)

concordia_network = ConcordiaNetwork(student_nn, teacher_psl, predicate_builder, lambda x: x, False)

concordia_network.fit(predicate_builder)