from __future__ import print_function

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
from utils import accumulate_gradient
from utils import update_graph_parameters
import random
import string
import time
import re
from collections import defaultdict
from tqdm import tqdm
import torch.utils.data as data


class EntityLinkingDataset(data.Dataset):
    def __init__(self, file_path, vocab, psl_predictions):
        self.dataset = TxtFolder_RNN(file_name=file_path, vocab=vocab)
        self.psl_predictions = psl_predictions

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        nn_data = collate_fn([self.dataset[index]])
        return (nn_data[0], nn_data[1], nn_data[2]), [self.psl_predictions[0][index]], None

    def _split_rating_data_to_xy(self, df_ratings):
        x = df_ratings[['user_id', 'item_id']].values
        y = df_ratings[['rating']].values
        return x, y


def concordia_data_loader(file_path, vocab, batch_size):
    dataset = TxtFolder_RNN(file_name=file_path, vocab=vocab)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # =int(self.opt.nThreads),
        drop_last=True,
        collate_fn=collate_fn
    )


def compute_dpl_loss(predictions, targets, args):
    class_weights = get_data_balancing_weights(targets, args)
    loss = F.kl_div(predictions, targets, reduction='none')
    loss = loss.sum(dim=1) * class_weights
    loss = loss.mean()
    return loss


def test(student, val_loader, args):
    student.model.eval()
    val_loss = 0
    predictions = []
    targets = []

    for data, batch_mask, mask, length, target in val_loader:
        if args.cuda:
            data, batch_mask, mask, target = data.cuda(), batch_mask.byte().cuda(), mask.cuda(), target.cuda()

        target = target.select(1, 1).contiguous().view(-1).long()
        output = student.predict((data, batch_mask, mask))[0]
        val_loss += F.nll_loss(output, target).item() / len(val_loader)
        predictions += np.exp(output.data[:, 1].cpu().numpy()).tolist()
        targets += target.data.cpu().tolist()

    recall, precision, f1, accuracy = compute_metrics(targets, predictions, threshold=0.5)
    print_log_metrics(val_loss, accuracy, precision, recall, f1)

    if args.tune_threshold:
        max_acc = accuracy
        for threshold in predictions:
            recall, precision, f1, accuracy = compute_metrics(targets, predictions, threshold=threshold)
            if accuracy > max_acc:
                max_acc = accuracy
                args.threshold = threshold
        print_log_metrics(val_loss, accuracy, precision, recall, f1)


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
    test_file_path = os.path.join(opt.dataroot, opt.test_data)

    train_loader = CreateDataLoader(opt.classifier_type,
                                    training_file_path,
                                    opt.vocab,
                                    opt.windowSize,
                                    opt.batch_size).load_data()

    val_loader = CreateDataLoader(opt.classifier_type,
                                  validation_file_path,
                                  opt.vocab,
                                  opt.windowSize,
                                  opt.batch_size).load_data()

    train_data = load_pickle_data(training_file_path)
    valid_data = load_pickle_data(validation_file_path)
    test_data = load_pickle_data(test_file_path)


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
    student = Student(model, None, optimizer)

    knowledge_base_factory = KnowledgeBaseFactory('Experiments/DPL/teacher/train')
    teacher_psl = PSLTeacher(predicates_to_infer=['z'],
                             knowledge_base_factory=knowledge_base_factory,
                             **config_concordia)
    psl_predictions = teacher_psl.predict(train_data)

    concordia = ConcordiaNetwork(student, teacher_psl, **concordia_config)

    train_data_loader = EntityLinkingDataset(training_file_path, vocab, psl_predictions)
    valid_data_loader = EntityLinkingDataset(validation_file_path, vocab, psl_predictions)

    concordia.fit(train_data_loader, valid_data_loader)
    pass


if __name__ == '__main__':
    options = load_arguments()
    sys.setrecursionlimit(20000)
    main(options)
