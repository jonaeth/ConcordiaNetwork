from __future__ import print_function
import torch.nn.functional as F
import torch.optim as optim
from NeuralNetworkModels.EncoderRNN import EncoderRNN
from data_loader import CreateDataLoader
import os
import sys
from load_arguments import load_arguments
from Concordia.Student import Student
from data_preparation import *
from metrics import *
from validation_utils import *
from visualizer import make_html_file, make_html_file_confidence
from utils import accumulate_gradient
from utils import update_graph_parameters
import random
import string
import time
import re
from collections import defaultdict
from tqdm import tqdm


def compute_dpl_loss(predictions, targets, args):
    class_weights = get_data_balancing_weights(targets, args)
    loss = F.kl_div(predictions, targets, reduction='none')
    loss = loss.sum(dim=1) * class_weights
    loss = loss.mean()
    return loss


# train procedure, we can use more complicated optimization method
def train_m_step_rnn(epoch, student, train_loader, args):
    student.model.train()
    for batch_idx, (data, batch_mask, mask, length, target) in enumerate(train_loader):
        if args.cuda:
            data, batch_mask, mask, target = data.cuda(), batch_mask.cuda().byte(), mask.cuda(), target.cuda()
        output = student.predict((data, batch_mask, mask))[0]
        loss = compute_dpl_loss(output, target, args)
        student.fit(loss)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))


def compute_predicates(data_instances):
    predicates = defaultdict(list)
    targets = []
    for trial_id, trial in enumerate(data_instances.keys()):
        for i in range(len(data_instances[trial]['inc'])):
            instance = data_instances[trial]['inc'][i]
            if not len(instance['pos_neg_example']):
                continue

            targets += [f"{trial_id}\t0\t{example_id}" for example_id in range(len(instance['pos_neg_example']))]
            for rv_name, rv_value in instance['graph'].base_assignment.items():
                example_id = re.findall('[0-9]+', rv_name)[0]
                predicate_name = rv_name[:rv_name.index(example_id)]
                predicates[predicate_name].append(f"{trial_id}\t0\t{example_id}\t{rv_value}")

    for trial_id, trial in enumerate(data_instances.keys()):
        for i in range(len(data_instances[trial]['exc'])):
            instance = data_instances[trial]['exc'][i]
            if not len(instance['pos_neg_example']):
                continue

            targets += [f"{trial_id}\t1\t{example_id}" for example_id in range(len(instance['pos_neg_example']))]
            for rv_name, rv_value in instance['graph'].base_assignment.items():
                example_id = re.findall('[0-9]+', rv_name)[0]
                predicate_name = rv_name[:rv_name.index(example_id)]
                predicates[predicate_name].append(f"{trial_id}\t1\t{example_id}\t{rv_value}")

    with open(f'Experiments/DPL/teacher/train/targets/z.psl', 'w') as f:
        for row in targets:
            f.write(f'{row}\n')

    for predicate_name, groundings in predicates.items():
        with open(f'Experiments/DPL/teacher/train/observations/{predicate_name}.psl', 'w') as f:
            for row in groundings:
                f.write(f'{row}\n')
    return predicates



def train_m_step_potential(result, args):
    # calculate the gradient over all the samples
    current_parameters = {}  # only need to get it one time
    grads = []

    for trial in result.keys():

        for i in range(len(result[trial]['inc'])):
            instance = result[trial]['inc'][i]
            if len(instance['pos_neg_example']) >= 1:
                current_parameters.update(
                    instance['graph'].get_current_parameters())  # reset the prior, this is necessary
                grads.append(instance['graph'].compute_gradient())  # get the gradient from this sample


        for i in range(len(result[trial]['exc'])):

            instance = result[trial]['exc'][i]

            # only care the specified entity type
            if len(instance['pos_neg_example']) >= 1:
                current_parameters.update(
                    instance['graph'].get_current_parameters())  # reset the prior, this is necessary
                grads.append(instance['graph'].compute_gradient())  # get the gradient from this sample

    # update the graph parameters given all the parameter key
    sample_gradient = accumulate_gradient(grads, current_parameters.keys())
    print("old parameters ")
    print(current_parameters)

    print("gradient ")
    print(sample_gradient)
    new_parameters = update_graph_parameters(current_parameters, sample_gradient, args.learn_rate_graph,
                                             args.graph_regularizer)

    print("new parameters ")
    print(new_parameters)

    # reset all the parameter for the factor graph
    for trial in result.keys():

        for i in range(len(result[trial]['inc'])):
            instance = result[trial]['inc'][i]

            if len(instance['pos_neg_example']) >= 1:
                instance['graph'].update_all_factor_graph_pairwise(new_parameters)  # reset the prior, this is necessary

            result[trial]['inc'][i] = instance

        for i in range(len(result[trial]['exc'])):

            instance = result[trial]['exc'][i]

            # only care the specified entity type
            if len(instance['pos_neg_example']) >= 1:
                instance['graph'].update_all_factor_graph_pairwise(
                    new_parameters)  # reset the prior, this is necessary

            result[trial]['exc'][i] = instance

    return result


def train_e_step(student, data, vocab, args):
    student.model.eval()

    result = deepcopy(data)
    # examples = {}
    # factor_cnt = 0

    for trial in tqdm(data.keys()):

        for i in range(len(data[trial]['inc'])):
            instance = data[trial]['inc'][i]
            new_instance = deepcopy(instance)
            input_data = map_text_to_vocab_indices(instance, vocab, args)

            # only care the specified entity type

            label = []
            potential = []

            for idx, item in enumerate(instance['pos_neg_example']):

                # vectorize the mask
                mask = torch.LongTensor([1 if item[0] <= i <= item[1] else 0 for i in range(len(new_instance['text']))])

                if args.cuda:
                    mask = mask.cuda()

                output = student.model.forward((input_data[None], None, mask[None]))[0]
                marginal = output.data.squeeze().cpu().numpy()  # get the index of the max log-probability
                # update each marginal probability for each mentions
                label.append(np.argmax(marginal))
                potential.append(marginal)

            if len(label) > 0:
                new_instance['graph'].set_label(label)
            for idx, item in enumerate(instance['pos_neg_example']):
                new_instance['graph'].update_factor_graph_unary("DL" + str(idx), potential[idx])

            result[trial]['inc'][i] = new_instance

        for i in range(len(data[trial]['exc'])):

            instance = data[trial]['exc'][i]
            new_instance = deepcopy(instance)
            input_data = map_text_to_vocab_indices(instance, vocab, args)

            # only care the specified entity type

            label = []
            potential = []
            for idx, item in enumerate(instance['pos_neg_example']):

                # vectorize the mask
                mask = torch.LongTensor([1 if item[0] <= i <= item[1] else 0 for i in range(len(new_instance['text']))])
                if args.cuda:
                    mask = mask.cuda()

                output = student.model.forward((input_data[None], None, mask[None]))[0]
                marginal = output.data.squeeze().cpu().numpy()  # get the index of the max log-probability
                label.append(np.argmax(marginal))
                potential.append(marginal)

            if len(label) > 0:
                new_instance['graph'].set_label(label)
            for idx, item in enumerate(instance['pos_neg_example']):
                new_instance['graph'].update_factor_graph_unary("DL" + str(idx), potential[idx])

            result[trial]['exc'][i] = new_instance

    return result


def distribute_message_passing(epoch, result):
    # now do the paralell message passing updating by writing all the files
    chunk = {}
    chunk_size = 200  # takes around 15 minutes on slurm, totally will result 100 cpu jobs on slurm
    idx = 1
    total_chunk = 0
    randomstring = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(30))
    random_dir = "./temp/%s" % randomstring
    if os.path.exists(random_dir):
        os.rmdir(random_dir)
    os.mkdir(random_dir)

    for trial in result.keys():
        chunk[trial] = result[trial]
        if idx % chunk_size == 0:
            input_file = os.path.join(random_dir, "mp_" + str(total_chunk) + "_" + str(epoch) + ".pkl")
            fp = open(input_file, "wb+")
            pickle.dump(chunk, fp)
            fp.close()
            # now call the shell to lanuch the job
            output_file = os.path.join(random_dir, "mp_" + str(total_chunk) + "_" + str(epoch + 1) + ".pkl")
            command = "bash submit_mp.sh %s %s " % (input_file, output_file)
            os.system(command)
            time.sleep(10)
            print("submitted job to slurm \n")
            chunk = {}
            total_chunk += 1

        idx += 1

    # the last chunks
    if len(chunk.keys()) > 0:
        input_file = os.path.join(random_dir, "mp_" + str(total_chunk) + "_" + str(epoch) + ".pkl")
        fp = open(input_file, "wb+")
        pickle.dump(chunk, fp)
        fp.close()
        # now call the shell to lanuch the job
        output_file = os.path.join(random_dir, "mp_" + str(total_chunk) + "_" + str(epoch + 1) + ".pkl")
        command = "bash submit_mp.sh %s %s " % (input_file, output_file)
        os.system(command)
        print("submitted job to slurm \n")
        total_chunk += 1

    print("total %d jobs submitted \n" % total_chunk)

    # checking all the new files generated
    while True:

        total_finished = 0
        for i in range(total_chunk):
            if not os.path.exists(os.path.join(random_dir, "mp_" + str(i) + "_" + str(epoch + 1) + ".pkl")):
                time.sleep(5)
            else:
                total_finished += 1

        if total_finished == total_chunk:
            cacheds = []
            new_state = []
            for i in range(total_chunk):
                cached = os.stat(os.path.join(random_dir, "mp_" + str(i) + "_" + str(epoch + 1) + ".pkl")).st_mtime
                cacheds.append(cached)
            time.sleep(100)
            for i in range(total_chunk):
                curr = os.stat(os.path.join(random_dir, "mp_" + str(i) + "_" + str(epoch + 1) + ".pkl")).st_mtime
                new_state.append(curr)

            if new_state == cacheds:
                break

    print("all message passing finished \n")
    # checking the result until get all of them, and then return the new marginal probability
    result = {}
    for i in range(total_chunk):
        fp = open(os.path.join(random_dir, "mp_" + str(i) + "_" + str(epoch + 1) + ".pkl"), "rb")
        chunk = pickle.load(fp)
        result.update(chunk)
        fp.close()

    return result


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


# test procedure
def write_validation_result_to_file(student, data, vocab, file_name, args):
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
def get_examples_and_confidences(student, data, vocab, entity_type, args):
    student.model.eval()
    results = get_results(data, student, entity_type, vocab, args)
    # prepration writing prediction to html file
    examples = get_examples(results)
    confidences = get_confidences_of_each_trial(results, entity_type, args)

    return examples, confidences


def main(opt):
    # init
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
    student = Student(model, compute_dpl_loss, optimizer)

    for epoch in range(1, opt.epochs + 1):

        # train the nn model
        if not opt.hard_em:

            if not opt.stochastic:

                # initial update the nn with all the examples
                if opt.stage == "M":

                    for k in range(opt.multiple_M):
                        train_m_step_rnn(epoch, student, train_loader, opt)
                        print(" threshold: %f \n" % opt.threshold)
                        # test after each epoch
                        test(student, val_loader, opt)
                        write_validation_result_to_file(student, valid_data, vocab, opt.prediction_file, opt)

                    # evaluate on the batch we sampled
                    test(student, val_loader, opt)
                    print(" threshold: %f \n" % opt.threshold)
                    write_validation_result_to_file(student, valid_data, vocab, opt.prediction_file, opt)

                    # save the model at each epoch, always use the newest one
                    torch.save(student.model.state_dict(), opt.save_path)

                else:  # only do the E step
                    # train the pairwise potential model
                    if opt.learn_graph:

                        # get the label for each instance, train_data is fresh
                        if result is None:
                            result = train_data
                        result = train_e_step(student, result, vocab, opt)
                        compute_predicates(train_data)

                        # train the graphical model with those labels
                        result = train_m_step_potential(result, opt)
                        # reset the prior, message-passing again
                        result = distribute_message_passing(epoch, result)

                        # save the data to disk so next time when readed, the marginal probability is changed
                        # slightly change the path to avoid re-write
                        # don't use it to save time for large dataset

                        file_path = os.path.join(opt.dataroot, "new_" + opt.train_data)
                        fp = open(file_path, "wb+")
                        pickle.dump(result, fp)  # save the new result to the disk
                        fp.close()

                        # directly read from the data
                        # args.file_path = result
                        # data_loader = CreateDataLoader(args)
                        # train_loader = data_loader.load_data()

        else:

            train_m_step_rnn(epoch, student, train_loader, opt)
            print(" threshold: %f \n" % opt.threshold)
            test(student, val_loader, opt)  # initial test
            write_validation_result_to_file(student, valid_data, vocab, opt.prediction_file, opt)
            # save the model at each epoch, always use the newest one
            torch.save(student.model.state_dict(), opt.save_path)

    # ignore this first
    test_result, confidences = get_examples_and_confidences(student, test_data, vocab, opt.entity_type, opt)

    # visualization the result using the visualizer
    print("writing the result to html \n")
    make_html_file(test_result, opt.visulization_html, opt.entity_type)

    # Load gene key data
    with open(opt.gene_key, 'rb') as f:
        gene_key = pickle.load(f)
        f.close()

    print("writing the confidence to html \n")
    make_html_file_confidence(confidences, opt.confidence_html, gene_key)

    # write the the confidence to file for calculating the precision and recall
    # make_csv_file(args.csv_file, confidence)

    # save the final model
    torch.save(student.model.state_dict(), opt.save_path)


if __name__ == '__main__':
    options = load_arguments()
    sys.setrecursionlimit(20000)
    main(options)
