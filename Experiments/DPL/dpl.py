from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
from NeuralNetworkModels.EncoderRNN import EncoderRNN
from data_loader import CreateDataLoader
import os
import pickle
import numpy as np
import sys
from load_arguments import load_arguments
from Concordia.Student import Student
from Concordia.ConcordiaNetwork import ConcordiaNetwork
from data_preparation import *
from metrics import *
from validation_utils import *

args = load_arguments()

sys.setrecursionlimit(20000)


def compute_dpl_loss(predictions, targets):
    class_weights = get_data_balancing_weights(targets)
    loss = F.kl_div(predictions, targets, reduction='none')
    loss = loss.sum(dim=1) * class_weights
    loss = loss.mean()
    return loss


set_initial_seed(args.seed, args)
wordvec = get_word2vec(args.word_embedding)
vocab = get_vocabulary_wrapper(args.vocab_path)

vocab_size = len(vocab)
print("vocab size:{}".format(vocab_size))
args.vocab = vocab


# dataset
training_file_path = os.path.join(args.dataroot, args.train_data)
validation_file_path = os.path.join(args.dataroot, args.val_data)
test_file_path = os.path.join(args.dataroot, args.test_data)

train_loader = CreateDataLoader(args.classifier_type,
                                training_file_path,
                                args.vocab,
                                args.windowSize,
                                args.batch_size).load_data()

val_loader = CreateDataLoader(args.classifier_type,
                              validation_file_path,
                              args.vocab,
                              args.windowSize,
                              args.batch_size).load_data()

train_data = load_pickle_data(training_file_path)
valid_data = load_pickle_data(validation_file_path)
test_data = load_pickle_data(test_file_path)

# model
model = EncoderRNN(args.embed_size,
                   args.hidden_size,
                   vocab_size,
                   args.num_layer,
                   args.cell,
                   wordvec,
                   args.class_label,
                   args.initial_model)

if args.cuda:
    model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
student = Student(model, compute_dpl_loss, optimizer)


# train procedure, we can use more complicated optimization method
def train_Mstep_RNN():
    student.model.train()
    for data, batch_mask, mask, length, target in train_loader:
        if args.cuda:
            data, batch_mask, mask, target = data.cuda(), batch_mask.cuda().byte(), mask.cuda(), target.cuda()
        output = student.predict((data, batch_mask, mask))[0]
        loss = compute_dpl_loss(output, target)
        student.fit(loss)


def test():
    student.model.eval()
    val_loss = 0
    predictions = []
    targets = []

    for data, batch_mask, mask, length, target in val_loader:
        if args.cuda:
            data, batch_mask, mask, target = data.cuda(), batch_mask.byte().cuda(), mask.cuda(), target.cuda()

        target = target.select(1, 1).contiguous().view(-1).long()
        output = student.predict((data, batch_mask, mask))[0]
        val_loss += F.nll_loss(output, target).item()/len(val_loader)
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


# test procedure
def GetResult_valid(data, vocab, file_name):
    student.model.eval()
    fp = open(file_name, "wt+")
    fp.write("threshold: %f \n" % args.threshold)

    for trial in data.keys():
        for i in range(len(data[trial]['inc'])):
            instance = data[trial]['inc'][i]
            input_data = map_text_to_vocab_indices(instance, vocab, args)

            for item in instance['pos_neg_example']:

                mask, batch_mask = create_masks(args, len(instance['text']), item[0], item[1])
                label, protein_probability, _ = predict_label(student, args, input_data, batch_mask, mask)
                write_prediction_to_file(file=fp,
                                         trial=trial,
                                         text=instance['text'],
                                         true_label=item[2],
                                         predicted_label=label,
                                         predicted_probability=np.exp(protein_probability))

    fp.close()


# test procedure
def GetResult(data, entity_type):
    student.model.eval()
    results = get_results(data, student, entity_type, vocab, args)
    # prepration writing prediction to html file
    examples = get_examples(results)
    confidences = get_confidences_of_each_trial(results, entity_type, args)

    return examples, confidences


for epoch in range(1, args.epochs + 1):

    # train the nn model
    if not args.hard_em:

        if not args.stochastic:

            # initial update the nn with all the examples
            if args.stage == "M":

                for k in range(args.multiple_M):
                    train_Mstep_RNN()
                    print(" threshold: %f \n" % args.threshold)
                    # test after each epoch
                    test()
                    GetResult_valid(valid_data, vocab, args.prediction_file)

                # evaluate on the batch we sampled
                test()
                print(" threshold: %f \n" % args.threshold)
                GetResult_valid(valid_data, vocab, args.prediction_file)

                # save the model at each epoch, always use the newest one
                torch.save(model.state_dict(), args.save_path)

    else:

        train_Mstep_RNN()
        print(" threshold: %f \n" % args.threshold)
        test()  # initial test
        GetResult_valid(valid_data, vocab, args.prediction_file)
        # save the model at each epoch, always use the newest one
        torch.save(model.state_dict(), args.save_path)

# ignore this first
test_result, confidence = GetResult(test_data, args.entity_type)

# visualization the result using the visualizer
print("writing the result to html \n")
# make_html_file(test_result, args.visulization_html, args.entity_type)  TODO

# Load gene key data
with open(args.gene_key, 'rb') as f:
    gene_key = pickle.load(f)
    f.close()

print("writing the confidence to html \n")
# make_html_file_confidence(confidence, args.confidence_html, gene_key)  TODO

# write the the confidence to file for calculating the precision and recall
# make_csv_file(args.csv_file, confidence)

# save the final model
torch.save(model.state_dict(), args.save_path)
