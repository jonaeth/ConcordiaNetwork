from __future__ import print_function

import sys

sys.path.append('.')

import sklearn.model_selection
import torch
import torch.nn.functional as F
from torch.nn import KLDivLoss
import torch.optim as optim
from NeuralNetworkModels.EncoderRNN import EncoderRNN
from data_loader import CreateDataLoader
import os
import sys
from load_arguments import load_arguments
from Concordia.Student import Student
from data_preparation import *
from config_concordia import config_concordia
from Concordia.Teacher import PSLTeacher
from Experiments.DPL.KnowledgeBaseFactory import KnowledgeBaseFactory
from metrics import *
from validation_utils import *
from Concordia.ConcordiaNetwork import ConcordiaNetwork
from config import neural_network_config, optimiser_config, concordia_config
from text_folder_rnn import TxtFolder_RNN
from rnn_data_loader import collate_fn
from visualizer import make_html_file, make_html_file_confidence
import torch.nn as nn
import torch.utils.data as data


class EntityLinkingDataset(data.Dataset):
    def __init__(self, file_path, vocab, psl_predictions, is_validation=False):
        self.dataset = TxtFolder_RNN(file_name=file_path, vocab=vocab)
        self.psl_predictions = psl_predictions
        self.is_validation = is_validation

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.is_validation:
            return self.dataset[index]
        return self.dataset[index], self.psl_predictions[0][index]


def main(opt):
    set_initial_seed(opt.seed, opt)
    wordvec = get_word2vec(opt.word_embedding)
    vocab = get_vocabulary_wrapper(opt.vocab_path)

    vocab_size = len(vocab)
    print("vocab size:{}".format(vocab_size))
    opt.vocab = vocab
    result = None
    # dataset
    training_file_path = os.path.join(opt.dataroot, opt.train_data)
    validation_file_path = os.path.join(opt.dataroot, opt.val_data)
    train_data = load_pickle_data(training_file_path)
    validation_data = load_pickle_data(validation_file_path)

    labeled_train_data, validation_data = sklearn.model_selection.train_test_split([(key, val) for key, val in validation_data.items()], test_size=0.5, random_state=2022)

    labeled_train_data = {key: val for key, val in labeled_train_data}
    validation_data = {key: val for key, val in validation_data}

    # model
    model = EncoderRNN(opt.embed_size,
                       opt.hidden_size,
                       vocab_size,
                       opt.num_layer,
                       opt.cell,
                       wordvec,
                       opt.class_label,
                       opt.initial_model)

    if opt.cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    def student_loss(student_predictions, targets):
        loss = nn.CrossEntropyLoss()
        return loss(student_predictions[0], targets)


    student = Student(model, student_loss, optimizer)

    knowledge_base_factory = KnowledgeBaseFactory('Experiments/DPL/teacher/train')
    teacher_psl = PSLTeacher(predicates_to_infer=['z'],
                             knowledge_base_factory=knowledge_base_factory,
                             **config_concordia)

    teacher_psl.fit(labeled_train_data, None)

    psl_predictions_unlabeled = teacher_psl.predict(train_data)
    psl_predictions_labeled = teacher_psl.predict(labeled_train_data)

    concordia = ConcordiaNetwork(student, teacher_psl, **concordia_config)

    train_data_loader_labeled = EntityLinkingDataset(labeled_train_data, vocab, psl_predictions_labeled)
    train_data_loader_unlabeled = EntityLinkingDataset(train_data, vocab, psl_predictions_unlabeled)
    valid_data_loader = EntityLinkingDataset(validation_data, vocab, None, is_validation=True)

    valid_data_loader = torch.utils.data.DataLoader(
        valid_data_loader,
        batch_size=256,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=collate_fn
    )

    train_data_loader_labeled = torch.utils.data.DataLoader(
        train_data_loader_labeled,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=collate_fn
    )

    train_data_loader_unlabeled = torch.utils.data.DataLoader(
        train_data_loader_unlabeled,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=collate_fn
    )

    concordia.fit_semisupervised(train_data_loader_unlabeled, train_data_loader_labeled, valid_data_loader, epochs=10, metrics={'f1_score': f1_score,
                                                                 'accuracy_score': accuracy_score,
                                                                 'recall_score': recall_score,
                                                                 'precision_score': precision_score})


if __name__ == '__main__':
    options = load_arguments()
    sys.setrecursionlimit(20000)
    main(options)
