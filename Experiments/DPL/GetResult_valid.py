def GetResult_valid(data, vocab, file_name):
    student.model.eval()
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

                output = model.forward((input_data[None], batch_mask[None].bool(), mask[None]))[0]
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
