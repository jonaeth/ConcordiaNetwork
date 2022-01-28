from __future__ import print_function

import sys

sys.path.append('')

import sklearn.model_selection
import torch.optim as optim
from NeuralNetworkModels.EncoderRNN import EncoderRNN
import os
import sys
import torch.nn as nn
from load_arguments import load_arguments
from Concordia.Student import Student
from data_preparation import *
from metrics import *
from validation_utils import *
from Concordia.ConcordiaNetwork import ConcordiaNetwork
from config import concordia_config
from text_folder_rnn import TxtFolder_RNN
from rnn_data_loader import collate_fn

import torch.utils.data as data


class EntityLinkingDataset(data.Dataset):
    def __init__(self, file_path, vocab, psl_predictions, is_validation=False):
        self.dataset = TxtFolder_RNN(file_name=file_path, vocab=vocab)
        self.psl_predictions = psl_predictions
        self.is_validation = is_validation

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def main(opt):
    set_initial_seed(opt.seed, opt)
    wordvec = get_word2vec(opt.word_embedding)
    vocab = get_vocabulary_wrapper(opt.vocab_path)

    vocab_size = len(vocab)
    print("vocab size:{}".format(vocab_size))
    opt.vocab = vocab
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

    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)

    def student_loss(student_predictions, targets):
        loss = nn.CrossEntropyLoss()
        return loss(student_predictions[0], targets)
    student = Student(model, student_loss, optimizer)



    concordia = ConcordiaNetwork(student, None, **concordia_config)

    train_data_loader_labeled = EntityLinkingDataset(labeled_train_data, vocab, None)
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





    concordia.fit_student_alone(train_data_loader_labeled, valid_data_loader, epochs=10, metrics={'f1_score': f1_score,
                                                                 'accuracy_score': accuracy_score,
                                                                 'recall_score': recall_score,
                                                                 'precision_score': precision_score})


if __name__ == '__main__':
    options = load_arguments()
    sys.setrecursionlimit(20000)
    main(options)
