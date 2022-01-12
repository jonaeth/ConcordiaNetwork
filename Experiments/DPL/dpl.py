from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
from NeuralNetworkModels.EncoderRNN import EncoderRNN
from data_loader import CreateDataLoader
import os
import pickle
from copy import deepcopy
import numpy as np
import sys
from load_arguments import load_arguments
from Concordia.Student import Student
from Concordia.ConcordiaNetwork import ConcordiaNetwork
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

args = load_arguments()

sys.setrecursionlimit(20000)

def set_initial_seed(seed, args):
    torch.manual_seed(seed)
    print(" use cuda: %d \n" % args.cuda)
    if args.cuda:
        torch.cuda.manual_seed(seed)

def get_word2vec(path_to_embeddings):
    if not path_to_embeddings:
        return None
    with open(path_to_embeddings, "rb") as fp:
        wordvec = pickle.load(fp)
    return wordvec


def get_vocabulary_wrapper(vocab_path):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab


def load_pickle_data(file_path):
    with open(file_path, "rb") as fp:
        data = pickle.load(fp)
    return data


def get_data_balancing_weights(target):
    # make the data balance
    num_pos = float(sum(target[:, 0] <= target[:, 1]))
    num_neg = float(sum(target[:, 0] > target[:, 1]))
    mask_pos = (target[:, 0] <= target[:, 1]).cpu().float()
    mask_neg = (target[:, 0] > target[:, 1]).cpu().float()
    weight = mask_pos * (num_pos + num_neg) / num_pos
    weight += mask_neg * (num_pos + num_neg) / num_neg
    if args.cuda:
        weight = weight.cuda()

    return weight


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


def compute_metrics(targets, predictions, threshold=0.5):
    predictions = np.where(np.exp(predictions) > threshold, 1, 0)
    recall = recall_score(targets, predictions)
    precision = precision_score(targets, predictions)
    f1 = f1_score(targets, predictions)
    accuracy = accuracy_score(targets, predictions)
    return recall, precision, f1, accuracy


def print_log_metrics(loss, accuracy, precision, recall, f1):
    print('\nVal set: Average loss: {:.4f}, Accuracy: {:.4f}%, precision: ({:.4f}), recall: ({:.4f}),'
          ' f1: ({:.4f}) \n'.format(loss, accuracy, precision, recall, f1))

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


def create_masks(len_text, start, end):
    start_pos = max([start - 10, 0])
    end_pos = min([end + 10, len_text])

    mask = torch.LongTensor([1 if start <= i <= end else 0 for i in range(len_text)])
    batch_mask = torch.LongTensor([0 if start_pos <= i < end_pos else 1 for i in range(len_text)])
    if args.cuda:
        mask = mask.cuda()
        batch_mask = batch_mask.byte().cuda()

    return mask, batch_mask


def predict_label(input_data, batch_mask, mask):
    output = model.forward((input_data[None], batch_mask[None].bool(), mask[None]))[0]
    pred = output.data.max(1)[1]  # get the index of the max log-probability
    prob = np.exp(output.data.max(1)[0].cpu().numpy()[0])

    label = pred.cpu().numpy()[0]

    if label == 0 and 1 - prob >= args.threshold:
        label = 1

    if label == 1 and prob < args.threshold:
        label = 0

    return label


def predict_label(input_data, batch_mask, mask):
    output = student.model.forward((input_data[None], batch_mask[None].bool(), mask[None]))[0]
    pred = output.data.max(1)[1]  # get the index of the max log-probability
    prob = np.exp(output.data.max(1)[0].cpu().numpy()[0])
    label = pred.cpu().numpy()[0]

    if label == 0 and 1 - prob >= args.threshold:
        label = 1

    if label == 1 and prob < args.threshold:
        label = 0
    prob = np.exp(output.data.cpu().numpy()[0][1])  # Prob(X=1)
    return label, prob


def map_text_to_vocab_indices(instance,vocab):
    text_inc = instance['text']
    tokens_inc = []
    tokens_inc.extend([vocab(token) for token in text_inc])
    tokens_inc = torch.LongTensor(tokens_inc)

    if args.cuda:
        tokens_inc = tokens_inc.cuda()
    return tokens_inc


def write_prediction_to_file(file, trial, text, true_label, predicted_label, predicted_probability):
    # write the prediction to file
    file.write(trial + "\t")
    file.write(" ".join(text) + "\t")
    file.write("true label:" + str(true_label) + "\t")
    file.write("prediction:" + str(predicted_label) + "\t")
    file.write("p(x=1):" + str(predicted_probability) + "\n")


# test procedure
def GetResult_valid(data, vocab, file_name):
    student.model.eval()
    fp = open(file_name, "wt+")
    fp.write("threshold: %f \n" % args.threshold)

    for trial in data.keys():
        for i in range(len(data[trial]['inc'])):
            instance = data[trial]['inc'][i]
            input_data = map_text_to_vocab_indices(instance, vocab)

            for item in instance['pos_neg_example']:

                mask, batch_mask = create_masks(len(instance['text']), item[0], item[1])
                label, prob = predict_label(input_data, batch_mask, mask)
                write_prediction_to_file(file=fp,
                                         trial=trial,
                                         text=instance['text'],
                                         true_label=item[2],
                                         predicted_label=label,
                                         predicted_probability=prob)

    fp.close()


# test procedure
def GetResult(data, entity_type, vocab):
    result = deepcopy(data)
    model.eval()
    examples = {}

    confidence = {}
    confidence['positive'] = []
    confidence['negative'] = []

    for trial in data.keys():

        for i in range(len(data[trial]['inc'])):
            instance = data[trial]['inc'][i]
            text_inc = instance['text']
            len_text = len(text_inc)
            # will resave the result
            new_instance = deepcopy(instance)
            new_instance['matched'][entity_type] = []

            tokens_inc = []
            tokens_inc.extend([vocab(token) for token in text_inc])
            tokens_inc = torch.LongTensor(tokens_inc)

            if args.cuda:
                tokens_inc = tokens_inc.cuda()
            # input_data = Variable(tokens_inc, volatile=True)    BELOW equivalent
            input_data = tokens_inc

            # only care the specified entity type
            checked_item = []
            for item in instance['matched'][entity_type]:

                if item in checked_item:
                    continue

                # add some filters here
                if len(" ".join(text_inc[item[0]:item[1] + 1])) <= 3:
                    continue

                checked_item.append(item)
                # vectorize the mask
                mask = [0 for k in range(len_text)]

                # need to change the mask settings, for entity's attention: -10:10 window size
                batch_mask = [1 for k in range(len_text)]  # ignore all first
                start_pos = max([item[0] - 10, 0])
                end_pos = min([item[1] + 10, len_text])
                batch_mask[start_pos:end_pos] = [0] * (end_pos - start_pos)

                for k in range(len(mask)):
                    if (k >= item[0] and k < item[1] + 1):
                        mask[k] = 1

                mask = torch.LongTensor(mask)
                batch_mask = torch.LongTensor(batch_mask)

                if args.cuda:
                    mask = mask.cuda()
                    batch_mask = batch_mask.byte().cuda()

                # mask = Variable(mask, volatile=True) Unnecessary

                output = model.forward((input_data[None], batch_mask[None], mask[None]))[0]
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                prob = np.exp(output.data.max(1)[0].cpu().numpy()[0])

                label = pred.cpu().numpy()[0]

                if label == 0 and 1 - prob >= args.threshold:
                    label = 1

                if label == 1 and prob < args.threshold:
                    label = 0

                # save the result to the data type
                new_instance['matched'][entity_type].append(
                    (item[0], item[1], label, np.exp(output.data.cpu().numpy())))

            result[trial]['inc'][i] = new_instance

        for i in range(len(data[trial]['exc'])):
            instance = data[trial]['exc'][i]
            text_exc = instance['text']
            len_text = len(text_exc)

            # will resave the result
            new_instance = deepcopy(instance)
            new_instance['matched'][entity_type] = []

            tokens_exc = []
            tokens_exc.extend([vocab(token) for token in text_exc])
            tokens_exc = torch.LongTensor(tokens_exc)

            if args.cuda:
                tokens_exc = tokens_exc.cuda()
            # input_data = Variable(tokens_exc, volatile=True) BELOW equivalent
            input_data = tokens_exc

            # only care the specified entity type
            checked_item = []
            for item in instance['matched'][entity_type]:

                if item in checked_item:
                    continue

                # add some filters here
                if len(" ".join(text_exc[item[0]:item[1] + 1])) <= 3:
                    continue

                checked_item.append(item)

                # vectorize the mask
                mask = [0 for k in range(len_text)]

                # need to change the mask settings, for entity's attention: -10:10 window size
                batch_mask = [1 for k in range(len_text)]  # ignore all first
                start_pos = max([item[0] - 10, 0])
                end_pos = min([item[1] + 10, len_text])
                batch_mask[start_pos:end_pos] = [0] * (end_pos - start_pos)

                for k in range(len(mask)):
                    if (k >= item[0]) and (k < item[1] + 1):
                        mask[k] = 1

                mask = torch.LongTensor(mask)
                batch_mask = torch.LongTensor(batch_mask)

                if args.cuda:
                    mask = mask.cuda()
                    batch_mask = batch_mask.byte().cuda()

                # mask = Variable(mask, volatile=True) Unnecessary

                output = model.forward((input_data[None], batch_mask[None], mask[None]))[0]
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                prob = np.exp(output.data.max(1)[0].cpu().numpy()[0])

                # save the result to the data type
                label = pred.cpu().numpy()[0]

                if label == 0 and 1 - prob >= args.threshold:
                    label = 1

                if label == 1 and prob < args.threshold:
                    label = 0

                new_instance['matched'][entity_type].append(
                    (item[0], item[1], label, np.exp(output.data.cpu().numpy())))

            result[trial]['exc'][i] = new_instance

    # prepration writing prediction to html file
    for trial in result.keys():

        examples[trial] = {}
        examples[trial]['inc'] = []
        examples[trial]['exc'] = []

        for instance in result[trial]['inc']:
            context_inc = " ".join(instance['text']).encode("ascii", "ignore").split()
            matches_inc = {}
            # matched for inc
            for key in instance['matched'].keys():
                matches_inc[key] = instance['matched'][key]

            examples[trial]['inc'].append((context_inc, matches_inc))

        for instance in result[trial]['exc']:
            context_exc = " ".join(instance['text']).encode("ascii", "ignore").split()
            matches_exc = {}
            # matched for inc
            for key in instance['matched'].keys():
                matches_exc[key] = instance['matched'][key]

            examples[trial]['exc'].append((context_exc, matches_exc))

    # sample the confidence example for evaluation
    positive = 0
    negative = 0

    for trial in result.keys():

        random_seed = True
        if np.random.randint(10) >= 5:
            random_seed = False

        # now checking each instance
        for instance in result[trial]['inc']:
            context_inc = " ".join(instance['text']).encode("ascii", "ignore").split()
            matches_inc = {}

            # matched for inc
            for key in instance['matched'].keys():
                matches_inc[key] = instance['matched'][key]

            if random_seed == True:
                # sub-sample some negative from inclusion examples
                for key in matches_inc.keys():
                    if key == entity_type:
                        for item in matches_inc[key]:
                            if item[2] == 0 and negative <= args.max_confidence_instance:
                                confidence['negative'].append(
                                    (trial, context_inc, item))  # remove all the other detection
                                negative += 1
                                # sub-sample some examples from exclusion examples
                for key in matches_inc.keys():
                    if key == entity_type:
                        for item in matches_inc[key]:
                            if item[2] == 1 and positive <= args.max_confidence_instance:
                                confidence['positive'].append(
                                    (trial, context_inc, item))  # remove all the other detection
                                positive += 1

        for instance in result[trial]['exc']:
            context_exc = " ".join(instance['text']).encode("ascii", "ignore").split()
            matches_exc = {}

            # matched for inc
            for key in instance['matched'].keys():
                matches_exc[key] = instance['matched'][key]

            # now choose the positive and negative example
            if random_seed == False:
                # sub-sample some negative from inclusion examples
                for key in matches_exc.keys():
                    if key == entity_type:
                        for item in matches_exc[key]:
                            if item[2] == 0 and negative <= args.max_confidence_instance:
                                confidence['negative'].append(
                                    (trial, context_exc, item))  # remove all the other detection
                                negative += 1

                # sub-sample some examples from exclusion examples
                for key in matches_exc.keys():
                    if key == entity_type:
                        for item in matches_exc[key]:
                            if item[2] == 1 and positive <= args.max_confidence_instance:
                                confidence['positive'].append(
                                    (trial, context_exc, item))  # remove all the other detection
                                positive += 1

    return examples, confidence


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
test_result, confidence = GetResult(test_data, args.entity_type, vocab)

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
