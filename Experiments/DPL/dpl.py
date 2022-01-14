from __future__ import print_function
import torch.nn.functional as F
import torch.optim as optim
from NeuralNetworkModels.EncoderRNN import EncoderRNN
from data_loader import CreateDataLoader
import os
import pickle
import sys
from load_arguments import load_arguments
from Concordia.Student import Student
from data_preparation import *
from metrics import *
from validation_utils import *
from visualizer import make_html_file, make_html_file_confidence


def compute_dpl_loss(predictions, targets, args):
    class_weights = get_data_balancing_weights(targets, args)
    loss = F.kl_div(predictions, targets, reduction='none')
    loss = loss.sum(dim=1) * class_weights
    loss = loss.mean()
    return loss


# train procedure, we can use more complicated optimization method
def train_m_step_rnn(student, train_loader, args):
    student.model.train()
    for data, batch_mask, mask, length, target in train_loader:
        if args.cuda:
            data, batch_mask, mask, target = data.cuda(), batch_mask.cuda().byte(), mask.cuda(), target.cuda()
        output = student.predict((data, batch_mask, mask))[0]
        loss = compute_dpl_loss(output, target, args)
        student.fit(loss)


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
                        train_m_step_rnn(student, train_loader, opt)
                        print(" threshold: %f \n" % opt.threshold)
                        # test after each epoch
                        test(student, val_loader, opt)
                        write_validation_result_to_file(student, valid_data, vocab, opt.prediction_file, opt)

                    # evaluate on the batch we sampled
                    test(student, val_loader, opt)
                    print(" threshold: %f \n" % opt.threshold)
                    write_validation_result_to_file(student, valid_data, vocab, opt.prediction_file, opt)

                    # save the model at each epoch, always use the newest one
                    torch.save(model.state_dict(), opt.save_path)

        else:

            train_m_step_rnn(student, train_loader, opt)
            print(" threshold: %f \n" % opt.threshold)
            test(student, val_loader, opt)  # initial test
            write_validation_result_to_file(student, valid_data, vocab, opt.prediction_file, opt)
            # save the model at each epoch, always use the newest one
            torch.save(model.state_dict(), opt.save_path)

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
    torch.save(model.state_dict(), opt.save_path)


if __name__ == '__main__':
    options = load_arguments()
    sys.setrecursionlimit(20000)
    main(options)
