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
