from __future__ import print_function

import sys

sys.path.append('.')

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
import joblib
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


def get_data_balancing_weights(predictions, target):
    # make the data balance
    num_pos = float(sum(target[:, 0] <= target[:, 1]))
    num_neg = float(sum(target[:, 0] > target[:, 1]))
    mask_pos = (target[:, 0] <= target[:, 1]).cpu().float()
    mask_neg = (target[:, 0] > target[:, 1]).cpu().float()
    weight = mask_pos * (num_pos + num_neg) / num_pos
    weight += mask_neg * (num_pos + num_neg) / num_neg
    return weight


def compute_dpl_loss(predictions, targets, args):
    kl_loss = torch.nn.KLDivLoss(reduction='none')
    kl_loss(F.log_softmax(predictions, dim=1), targets)

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
    print('W2V and vocab is loaded')
    vocab_size = len(vocab)
    print("vocab size:{}".format(vocab_size))
    opt.vocab = vocab
    result = None
    # dataset

    training_file_path = os.path.join(opt.dataroot, opt.train_data)
    validation_file_path = os.path.join(opt.dataroot, opt.val_data)
    print('Loading training data')

    #train_data = load_pickle_data(training_file_path)
    valid_data = load_pickle_data(validation_file_path)
    print('Train data is loaded and vocab is loaded')


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
    student = Student(model, None, optimizer)

    knowledge_base_factory = KnowledgeBaseFactory('Experiments/DPL/teacher/train')
    teacher_psl = PSLTeacher(predicates_to_infer=['z'],
                             knowledge_base_factory=knowledge_base_factory,
                             **config_concordia)

    concordia = ConcordiaNetwork(student, teacher_psl, **concordia_config)

    psl_predictions = teacher_psl.predict(valid_data)


    train_data_loader = EntityLinkingDataset(validation_file_path, vocab, psl_predictions)
    valid_data_loader = EntityLinkingDataset(validation_file_path, vocab, psl_predictions, is_validation=True)



    valid_data_loader = torch.utils.data.DataLoader(
        valid_data_loader,
        batch_size=256,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=collate_fn
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_data_loader,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=collate_fn
    )


    concordia.fit_unsupervised(train_data_loader, valid_data_loader, epochs=10, metrics={'f1_score': f1_score,
                                                                    'accuracy_score': accuracy_score,
                                                                     'recall_score': recall_score,
                                                                     'precision_score': precision_score})

    concordia.predict(valid_data_loader)


if __name__ == '__main__':
    options = load_arguments()
    sys.setrecursionlimit(20000)
    main(options)
