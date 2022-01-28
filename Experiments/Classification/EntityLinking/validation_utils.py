import torch
import numpy as np
from copy import deepcopy


def create_masks(args, len_text, start, end):
    start_pos = max([start - 10, 0])
    end_pos = min([end + 10, len_text])

    mask = torch.LongTensor([1 if start <= i <= end else 0 for i in range(len_text)])
    batch_mask = torch.LongTensor([0 if start_pos <= i < end_pos else 1 for i in range(len_text)])
    if args.cuda:
        mask = mask.cuda()
        batch_mask = batch_mask.byte().cuda()

    return mask, batch_mask


def predict_label(student, args, input_data, batch_mask, mask):
    output = student.model.forward((input_data[None], batch_mask[None].bool(), mask[None]))[0]
    pred = output.data.max(1)[1]  # get the index of the max log-probability
    prob = np.exp(output.data.max(1)[0].cpu().numpy()[0])
    label = pred.cpu().numpy()[0]

    if label == 0 and 1 - prob >= args.threshold:
        label = 1

    if label == 1 and prob < args.threshold:
        label = 0

    log_probabilities = output.data.cpu().numpy()
    protein_log_probability = log_probabilities[0][1]  # Prob(X=1)
    return label, protein_log_probability, log_probabilities


def map_text_to_vocab_indices(instance, vocab, args):
    text = instance['text']
    tokens = []
    tokens.extend([vocab(token) for token in text])
    tokens = torch.LongTensor(tokens)

    if args.cuda:
        tokens = tokens.cuda()
    return tokens


def write_prediction_to_file(file, trial, text, true_label, predicted_label, predicted_probability):
    # write the prediction to file
    file.write(trial + "\t")
    file.write(" ".join(text) + "\t")
    file.write("true label:" + str(true_label) + "\t")
    file.write("prediction:" + str(predicted_label) + "\t")
    file.write("p(x=1):" + str(predicted_probability) + "\n")


def get_context_and_matches(instance):
    context = " ".join(instance['text']).encode("ascii", "ignore").split()
    matches = {}
    # matched for inc
    for key in instance['matched'].keys():
        matches[key] = instance['matched'][key]

    return context, matches


def get_examples(result):
    examples = {}
    for trial in result.keys():
        examples[trial] = {}
        examples[trial]['inc'] = []
        examples[trial]['exc'] = []

        for instance in result[trial]['inc']:
            context_inc, matches_inc = get_context_and_matches(instance)
            examples[trial]['inc'].append((context_inc, matches_inc))

        for instance in result[trial]['exc']:
            context_exc, matches_exc = get_context_and_matches(instance)
            examples[trial]['exc'].append((context_exc, matches_exc))

    return examples


def sample_matches(matches, trial, context, entity_type, positive, negative, confidence, args):
    for key in matches.keys():
        if key == entity_type:
            for item in matches[key]:
                if item[2] == 0 and negative <= args.max_confidence_instance:
                    confidence['negative'].append(
                        (trial, context, item))  # remove all the other detection
                    negative += 1
                    # sub-sample some examples from exclusion examples
                if item[2] == 1 and positive <= args.max_confidence_instance:
                    confidence['positive'].append(
                        (trial, context, item))  # remove all the other detection
                    positive += 1


def get_result_of_trial(instance, student, entity_type, vocab, args):
    input_data = map_text_to_vocab_indices(instance, vocab, args)
    # will resave the result
    new_instance = deepcopy(instance)
    new_instance['matched'][entity_type] = []

    # only care the specified entity type
    checked_item = []
    for item in instance['matched'][entity_type]:

        if item in checked_item:
            continue

        # add some filters here
        if len(" ".join(instance['text'][item[0]:item[1] + 1])) <= 3:
            continue

        checked_item.append(item)

        mask, batch_mask = create_masks(args, len(instance['text']), item[0], item[1])

        label, _, log_probabilites = predict_label(student, args, input_data, batch_mask, mask)

        # save the result to the data type
        new_instance['matched'][entity_type].append(
            (item[0], item[1], label, np.exp(log_probabilites)))

    return new_instance


def get_results(data, student, entity_type, vocab, args):
    result = deepcopy(data)

    for trial in data.keys():

        for i in range(len(data[trial]['inc'])):
            instance = data[trial]['inc'][i]
            result[trial]['inc'][i] = get_result_of_trial(instance, student, entity_type, vocab, args)

        for i in range(len(data[trial]['exc'])):
            instance = data[trial]['exc'][i]
            result[trial]['exc'][i] = get_result_of_trial(instance, student, entity_type, vocab, args)

    return result


def get_confidence_of_trial(instance, trial, entity_type, positive, negative, args):
    context = " ".join(instance['text']).encode("ascii", "ignore").split()
    matches = {}

    # matched for inc
    for key in instance['matched'].keys():
        matches[key] = instance['matched'][key]

    negative_confidences = []
    positive_confidences = []
    # sub-sample some negative from inclusion examples
    for key in matches.keys():
        if key == entity_type:
            for item in matches[key]:
                if item[2] == 0 and negative <= args.max_confidence_instance:
                    negative_confidences.append(
                        (trial, context, item))  # remove all the other detection
                    negative += 1
                    # sub-sample some examples from exclusion examples
                if item[2] == 1 and positive <= args.max_confidence_instance:
                    positive_confidences.append(
                        (trial, context, item))  # remove all the other detection
                    positive += 1

    return negative_confidences, positive_confidences


def get_confidences_of_each_trial(results, entity_type, args):
    positive_confidences = []
    negative_confidences = []

    for trial in results.keys():
        random_seed = False if np.random.randint(10) >= 5 else True
        positive = len(positive_confidences)
        negative = len(negative_confidences)
        # now checking each instance
        if random_seed:
            for instance in results[trial]['inc']:
                trial_positive_confidences, trial_negative_confidences = get_confidence_of_trial(instance,
                                                                                                 trial,
                                                                                                 entity_type,
                                                                                                 positive,
                                                                                                 negative,
                                                                                                 args)
                positive_confidences += trial_positive_confidences
                negative_confidences += trial_negative_confidences
        else:
            for instance in results[trial]['exc']:
                # now choose the positive and negative example
                trial_positive_confidences, trial_negative_confidences = get_confidence_of_trial(instance,
                                                                                                 trial,
                                                                                                 entity_type,
                                                                                                 positive,
                                                                                                 negative,
                                                                                                 args)
                positive_confidences += trial_positive_confidences
                negative_confidences += trial_negative_confidences

    return {'positive': positive_confidences, 'negative': negative_confidences}
