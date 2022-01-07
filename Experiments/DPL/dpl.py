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

args = load_arguments()

sys.setrecursionlimit(20000)

torch.manual_seed(args.seed)
print(" use cuda: %d \n" % args.cuda)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# we can specify the model here later
wordvec = None
if args.word_embedding:
    fp = open(args.word_embedding, "rb")
    wordvec = pickle.load(fp)
    fp.close()

# Load vocabulary wrapper
with open(args.vocab_path, 'rb') as f:
    vocab = pickle.load(f)
    f.close()

vocab_size = len(vocab)
print("vocab size:{}".format(vocab_size))
args.vocab = vocab

# dataset
args.file_path = os.path.join(args.dataroot, args.train_data)
data_loader = CreateDataLoader(args)
train_loader = data_loader.load_data()
args.file_path = os.path.join(args.dataroot, args.val_data)
data_loader = CreateDataLoader(args)
val_loader = data_loader.load_data()

# re-evaluate the probability under current rnn model
fp = open(os.path.join(args.dataroot, args.train_data), "rb")
train_data = pickle.load(fp)
fp.close()

fp = open(os.path.join(args.dataroot, args.val_data), "rb")
valid_data = pickle.load(fp)
fp.close()

# for visualization
args.file_path = os.path.join(args.dataroot, args.test_data)
with open(args.file_path, "rb") as fp:
    test_data = pickle.load(fp)
# fp = open(args.file_path)
# test_data = pickle.load(fp)
# fp.close()

# model
model = EncoderRNN(args.embed_size, args.hidden_size, vocab_size, args.num_layer, args.cell, wordvec, args.class_label)

if args.cuda:
    model.cuda()

# load the pre-trained model
if args.initial_model and os.path.exists(args.initial_model):
    print("load pre-trained model")
    model.load_state_dict(torch.load(args.initial_model))
    if args.cuda:
        model = model.cuda()

if not args.fix_embed:
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


# print ("remove all previous message passing \n")
# command = "rm ./result/mp_*.pkl"
# os.system(command)

# train procedure, we can use more complicated optimization method
def train_Mstep_RNN(epoch):
    model.train()
    all_instance = []

    for batch_idx, (data, batch_mask, mask, length, target) in enumerate(train_loader):

        if args.cuda:
            data, batch_mask, mask, target = data.cuda(), batch_mask.cuda().byte(), mask.cuda(), target.cuda()

        # make the data balance
        num_pos = float(sum(target[:, 0] <= target[:, 1]))
        num_neg = float(sum(target[:, 0] > target[:, 1]))
        weight = torch.ones(target.size(0), 1)
        mask_pos = (target[:, 0] <= target[:, 1]).cpu().float()
        mask_neg = (target[:, 0] > target[:, 1]).cpu().float()
        weight = mask_pos * (num_pos + num_neg) / num_pos
        weight += mask_neg * (num_pos + num_neg) / num_neg
        if args.cuda:
            weight = weight.cuda()
            # weight = Variable(weight).cuda() Unnecessary

        # data, mask, target = Variable(data), Variable(mask), Variable(target) NOW USELESS in new pytorch
        optimizer.zero_grad()
        output = model.forward(data, batch_mask, mask)
        loss = F.kl_div(output, target, reduction='none')
        loss = loss.sum(dim=1) * weight
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        print('train_Mstep_RNN')
        # if batch_idx % args.log_interval == 0:
        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #        epoch, batch_idx * len(data), len(train_loader.dataset),
        #        100. * batch_idx / len(train_loader), loss.data[0]))


def test(epoch):
    model.eval()
    val_loss = 0
    correct = 0
    prob_pos = []
    prob_neg = []

    for data, batch_mask, mask, length, target in val_loader:

        if args.cuda:
            data, batch_mask, mask, target = data.cuda(), batch_mask.byte().cuda(), mask.cuda(), target.cuda()

        # data, mask, target = Variable(data, volatile=True), Variable(mask, volatile=True), Variable(target.select(1, 1).contiguous().view(-1).long())
        # FIXED below, volatile=True should be equivalent to requires_grad=False, which is the default
        # Variable and Tensor have been merged
        target = target.select(1, 1).contiguous().view(-1).long()

        output = model.forward(data, batch_mask, mask)

        val_loss += F.nll_loss(output, target).item()  # need to check here
        pred = output.data.max(1)[1]  # get the index of the max log-probability

        pos_prob = [np.exp(output.data.cpu().numpy()[idx][1]) for idx, l in enumerate(target.data.cpu().numpy()) if
                    l == 1]
        neg_prob = [np.exp(output.data.cpu().numpy()[idx][1]) for idx, l in enumerate(target.data.cpu().numpy()) if
                    l == 0]

        prob_pos.extend(pos_prob)
        prob_neg.extend(neg_prob)

        correct += pred.eq(target.data).cpu().sum()

    # adjust the threhold according to the sampled label
    all_prob = deepcopy(prob_pos)
    all_prob.extend(prob_neg)
    all_prob = np.asarray(all_prob)
    prob_pos = np.asarray(prob_pos)
    prob_neg = np.asarray(prob_neg)

    # 0.5 is the standard threshold
    tp = np.sum(prob_pos >= 0.5)
    fp = np.sum(prob_pos < 0.5)
    tn = np.sum(prob_neg < 0.5)
    fn = np.sum(prob_neg >= 0.5)
    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)
    f1 = 2.0 * precision * recall / (precision + recall)

    val_loss = val_loss
    val_loss /= len(val_loader)  # loss function already averages over batch size
    print(
        '\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%), precision: ({:.4f}), recall: ({:.4f}), f1: ({:.4f}) \n'.format(
            val_loss, correct, len(val_loader.dataset),
            100. * correct / len(val_loader.dataset), precision, recall, f1))

    max_acc = float(correct) / len(val_loader.dataset)

    for threshold in all_prob:

        tp = np.sum(prob_pos >= threshold)
        fp = np.sum(prob_pos < threshold)
        tn = np.sum(prob_neg < threshold)
        fn = np.sum(prob_neg >= threshold)
        acc = (tp + tn) / float(len(all_prob))

        if acc > max_acc:
            max_acc = acc
            args.threshold = threshold
            precision = tp / float(tp + fp)
            recall = tp / float(tp + fn)
            f1 = 2.0 * precision * recall / (precision + recall)

    # print('\nVal set: Average loss: {:.4f}, Adjusted Accuracy: {}/{} ({:.4f}%), precision: ({:.4f}), recall: ({:.4f}), f1: ({:.4f})\n'.format(
    #    val_loss, len(val_loader.dataset)*max_acc, len(val_loader.dataset),
    #    100. * max_acc, precision, recall, f1))


# test procedure
def GetResult_valid(data, vocab, file_name):
    model.eval()
    fp = open(file_name, "wt+")
    fp.write("threshold: %f \n" % args.threshold)

    for trial in data.keys():
        for i in range(len(data[trial]['inc'])):
            instance = data[trial]['inc'][i]
            text_inc = instance['text']
            len_text = len(text_inc)
            tokens_inc = []
            tokens_inc.extend([vocab(token) for token in text_inc])
            tokens_inc = torch.LongTensor(tokens_inc)

            if args.cuda:
                tokens_inc = tokens_inc.cuda()
            # input_data = Variable(tokens_inc, volatile=True)    BELOW is Equivalent in new torch version
            input_data = tokens_inc

            for item in instance['pos_neg_example']:

                # vectorize the mask, for the entity
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

                # mask = Variable(mask, volatile=True) Unnecessary in new torch version

                output = model.forward(input_data[None], batch_mask[None].bool(), mask[None])
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                prob = np.exp(output.data.max(1)[0].cpu().numpy()[0])

                label = pred.cpu().numpy()[0]

                if label == 0 and 1 - prob >= args.threshold:
                    label = 1

                if label == 1 and prob < args.threshold:
                    label = 0

                # write the prediction to file
                fp.write(trial + "\t")
                fp.write(" ".join(text_inc) + "\t")
                fp.write("true label:" + str(item[2]) + "\t")
                fp.write("prediction:" + str(label) + "\t")
                fp.write("p(x=1):" + str(np.exp(output.data.cpu().numpy()[0][1])) + "\n")

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

                output = model.forward(input_data[None], batch_mask[None], mask[None])
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

                output = model.forward(input_data[None], batch_mask[None], mask[None])
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
                    train_Mstep_RNN(epoch)
                    print(" threshold: %f \n" % args.threshold)
                    # test after each epoch
                    test(epoch)
                    GetResult_valid(valid_data, vocab, args.prediction_file)

                # evaluate on the batch we sampled
                test(epoch)
                print(" threshold: %f \n" % args.threshold)
                GetResult_valid(valid_data, vocab, args.prediction_file)

                # save the model at each epoch, always use the newest one
                torch.save(model.state_dict(), args.save_path)

    else:

        train_Mstep_RNN(epoch)
        print(" threshold: %f \n" % args.threshold)
        test(epoch)  # initial test
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
