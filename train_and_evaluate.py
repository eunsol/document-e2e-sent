import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import autograd
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from random import shuffle
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import graph_results as plotter
import data_processor as parser
from advanced_model_1 import Model1
#  from advanced_model_2 import Model2

NUM_LABELS = 3
# convention: [NEG, NULL, POS]
epochs = 10
EMBEDDING_DIM = 50
MAX_CO_OCCURS = 10
MAX_NUM_MENTIONS = 10
HIDDEN_DIM = EMBEDDING_DIM
NUM_POLARITIES = 6
DROPOUT_RATE = 0.2
using_GPU = torch.cuda.is_available()
threshold = torch.log(torch.FloatTensor([0.5, 0.1, 0.5]))
if using_GPU:
    threshold = threshold.cuda()
MODEL = Model1

set_name = "F"
datasets = {"A": {"filepath": "./data/new_annot/feature",
                  "filenames": ["new_train.json", "acl_dev_eval_new.json", "acl_test_new.json"],
                  "weights": torch.FloatTensor([0.8, 1.825, 1]),
                  "batch": 10},
            "B": {"filepath": "./data/new_annot/trainsplit_holdtarg",
                  "filenames": ["train.json", "dev.json", "test.json"],
                  "weights": torch.FloatTensor([0.77, 1.766, 1]),
                  "batch": 10},
            "C": {"filepath": "./data/new_annot/feature",
                  "filenames": ["train_90_null.json", "acl_dev_eval_new.json", "acl_test_new.json"],
                  "weights": torch.FloatTensor([1, 0.07, 1.26]),
                  "batch": 50},
            "D": {"filepath": "./data/new_annot/feature",
                  "filenames": ["acl_dev_tune_new.json", "acl_dev_eval_new.json", "acl_test_new.json"],
                  "weights": torch.FloatTensor([2.7, 0.1, 1]),
                  "batch": 10},
            "E": {"filepath": "./data/new_annot/feature",
                  "filenames": ["E_train.json", "acl_dev_eval_new.json", "acl_test_new.json"],
                  "weights": torch.FloatTensor([1, 0.3523, 1.0055]),
                  "batch": 25},
            "F": {"filepath": "./data/new_annot/feature",
                  "filenames": ["F_train.json", "acl_dev_eval_new.json", "acl_test_new.json"],
                  "weights": torch.FloatTensor([1, 0.054569, 1.0055]),
                  "batch": 80},
            "G": {"filepath": "./data/new_annot/feature",
                  "filenames": ["G_train.json", "acl_dev_eval_new.json", "acl_test_new.json"],
                  "weights": torch.FloatTensor([1.823, 0.0699, 1.0055]),
                  "batch": 100},
            "H": {"filepath": "./data/new_annot/feature",
                  "filenames": ["H_train.json", "acl_dev_eval_new.json", "acl_test_new.json"],
                  "weights": torch.FloatTensor([1, 0.054566, 1.0055]),
                  "batch": 100},
            "I": {"filepath": "./data/new_annot/mpqa_split",
                  "filenames": ["train.json", "dev.json", "test.json"],
                  "weights": torch.FloatTensor([1.3745, 0.077, 1]),
                  "batch": 50}
            }

BATCH_SIZE = datasets[set_name]["batch"]

def train(Xtrain, Xdev, Xtest,
          model, word_to_ix, ix_to_word,
          using_GPU, lr_decay=1):
    print("Evaluating before training...")
    train_res = []
    dev_res = []
    dev_f1_aves = []
    test_res = []
    train_accs = []
    dev_accs = []
    test_accs = []

    weights = datasets[set_name]["weights"]
    if using_GPU:
        weights = weights.cuda()
    loss_function = nn.NLLLoss(weight=weights)
    train_loss_epoch = []
    dev_loss_epoch = []

    '''
    # Just for 1 batch
    for batch in Xtrain:
        plotter.graph_attention(model, word_to_ix, ix_to_word, batch, using_GPU)
        break
    '''
    print("evaluating training...")
    train_score, train_acc = evaluate(model, word_to_ix, ix_to_word, Xtrain, using_GPU)
    print("train f1 scores = " + str(train_score))
    dev_score, dev_acc = evaluate(model, word_to_ix, ix_to_word, Xdev, using_GPU)
    print("dev f1 scores = " + str(dev_score))
    train_res.append(train_score)
    dev_res.append(dev_score)
    dev_f1_aves.append(sum(dev_score) / len(dev_score))
    best_epoch = 0
    train_accs.append(train_acc)
    dev_accs.append(dev_acc)

    test_score, test_acc = evaluate(model, word_to_ix, ix_to_word, Xtest, using_GPU)
    test_res.append(test_score)
    test_accs.append(test_acc)

    # skip updating the non-requires-grad params (i.e. the embeddings)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=0)

    for epoch in range(0, epochs):
        losses = []
        print("Epoch " + str(epoch))
        i = 0
        for batch in Xtrain:
            (words, lengths), polarity, holder_target, label = batch.text, batch.polarity, batch.holder_target, batch.label
            (holders, holder_lengths) = batch.holder_index
            (targets, target_lengths) = batch.target_index
            co_occur_feature = batch.co_occurrences
            holder_rank, target_rank = batch.holder_rank, batch.target_rank

            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
            model.batch_size = len(label.data)  # set batch size

            # Step 3. Run our forward pass.
            log_probs = model(words, polarity, holder_target, lengths,
                              holders, targets, holder_lengths, target_lengths,
                              co_occur_feature=co_occur_feature,
                              holder_rank=holder_rank, target_rank=target_rank)  # log probs: batch_size x 3

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss = loss_function(log_probs, label)  # log_probs = actual distr, target = computed distr
            losses.append(float(loss))
            loss.backward()
            optimizer.step()
            # print("loss = " + str(loss))
            if (i % 10 == 0):
                print("    " + str(i))
            i += 1
        print("loss = " + str((sum(losses) / len(losses))))
        train_loss_epoch.append(float(sum(losses)) / float(len(losses)))
        # Apply decay
        if epoch % 10 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay
        '''
        # Just for 1 batch
        for batch in Xtrain:
            plotter_graph_attention(model, word_to_ix, ix_to_word, batch, using_GPU)
            break
        '''
        print("Evaluating...")
        print("evaluating training...")
        train_score, train_acc = evaluate(model, word_to_ix, ix_to_word, Xtrain, using_GPU)
        print("train f1 scores = " + str(train_score))
        print(Xdev)
        dev_score, dev_acc = evaluate(model, word_to_ix, ix_to_word, Xdev, using_GPU,
                                      losses=dev_loss_epoch, loss_fxn=loss_function)
        print("dev loss = " + str(dev_loss_epoch[len(dev_loss_epoch) - 1]))
        print("dev f1 scores = " + str(dev_score))
        train_res.append(train_score)
        train_accs.append(train_acc)
        dev_res.append(dev_score)
        dev_f1_aves.append(sum(dev_score) / len(dev_score))
        if dev_f1_aves[epoch] > dev_f1_aves[best_epoch]:
            best_epoch = epoch
            print("Updated best epoch: " + str(dev_f1_aves[best_epoch]))
        dev_accs.append(dev_acc)
        test_score, test_acc = evaluate(model, word_to_ix, ix_to_word, Xtest, using_GPU)
        test_res.append(test_score)
        test_accs.append(test_acc)
    print("dev losses:")
    print(dev_loss_epoch)
    return train_res, dev_res, test_res, train_accs, dev_accs, test_accs, train_loss_epoch, best_epoch


def decode(word_indices, ix_to_word):
    words = [ix_to_word[index] for index in word_indices.data]
    return words


def evaluate(model, word_to_ix, ix_to_word, Xs, using_GPU,
             losses=None, loss_fxn=None):
    # Set model to eval mode to turn off dropout.
    model.eval()

    total_true = [0, 0, 0]
    total_pred = [0, 0, 0]
    total_correct = [0, 0, 0]

    num_examples = 0
    num_correct = 0

    print("Iterate across : " + str(len(Xs)) + " batch(es)")
    counter = 0
    loss_this_batch = []
    # count positive classifications in pos
    for batch in Xs:
        counter += 1
        # print(word_to_ix)
        (words, lengths), polarity, holder_target, label = batch.text, batch.polarity, batch.holder_target, batch.label
        (holders, holder_lengths) = batch.holder_index
        (targets, target_lengths) = batch.target_index
        co_occur_feature = batch.co_occurrences
        holder_rank, target_rank = batch.holder_rank, batch.target_rank

        # words.no_grad() = lengths.no_grad() = polarity.no_grad() = label.no_grad() = True
        # holders.no_grad() = targets.no_grad() = holder_lengths.no_grad() = target_lengths.no_grad() = True

        model.batch_size = len(label.data)  # set batch size
        '''
        if len(label.data) > BATCH_SIZE:
            print(label.data)
        '''
        log_probs = model(words, polarity, holder_target, lengths,
                          holders, targets, holder_lengths, target_lengths,
                          co_occur_feature=co_occur_feature,
                          holder_rank=holder_rank, target_rank=target_rank)  # log probs: batch_size x 3
        if losses is not None:
            loss = loss_fxn(log_probs, label)
            loss_this_batch.append(float(loss))

        # '''
        pred_label = log_probs.data.max(1)[1]  # torch.ones(len(log_probs), dtype=torch.long)
        '''
        pred_label = torch.ones(len(log_probs), dtype=torch.long)
        if using_GPU:
            pred_label = pred_label.cuda()
        pred_label[log_probs[:, 2] > log_probs[:, 0]] = 2  # max of the 2
        pred_label[log_probs[:, 0] > log_probs[:, 2]] = 0
        pred_label[log_probs[:, 1] > threshold[1]] = 1  # predict is 1 if even just > 10% certainty
        '''

        # Count the number of examples in this batch
        for i in range(0, NUM_LABELS):
            total_true[i] += torch.sum(label.data == i)
            total_pred[i] += torch.sum(pred_label == i)
            total_correct[i] += torch.sum((pred_label == i) * (label.data == i))
            '''
            print(i)
            print(label.data)
            print(pred_label)
            print(total_correct)
            '''
        num_correct += float(torch.sum(pred_label == label.data))
        num_examples += len(label.data)

        assert sum(total_true) == num_examples
        assert sum(total_pred) == num_examples
        assert sum(total_correct) == num_correct

        if counter % 50 == 0:
            print(counter)

    if losses is not None:
        losses.append(sum(loss_this_batch) / len(loss_this_batch))

    # Compute f1 scores (separate method?)
    precision = [0, 0, 0]
    recall = [0, 0, 0]
    f1 = [0, 0, 0]
    for i in range(0, NUM_LABELS):
        if total_pred[i] == 0:
            precision[i] = 0.0
        else:
            precision[i] = float(total_correct[i]) / float(total_pred[i])
        recall[i] = float(total_correct[i]) / float(total_true[i])
        if precision[i] + recall[i] == 0:
            f1[i] = 0.0
        else:
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

    # Compute accuracy
    accuracy = num_correct / float(num_examples)
    print(accuracy)
    print("precision: " + str(precision))
    print("recall: " + str(recall))

    print(f1)
    #    score = f1_score(list(predictions), list(truths), labels=[0, 1, 2], average=None)
    #    print(score)

    # Set the model back to train mode, to reactivate dropout.
    model.train()

    return f1, accuracy


def main():
    train_data, dev_data, test_data, TEXT, DOCID, POLARITY = parser.parse_input_files(BATCH_SIZE, EMBEDDING_DIM,
                                                                                      using_GPU,
                                                                                      filepath=datasets[set_name][
                                                                                          "filepath"],
                                                                                      train_name=
                                                                                      datasets[set_name]["filenames"][
                                                                                          0],
                                                                                      dev_name=
                                                                                      datasets[set_name]["filenames"][
                                                                                          1],
                                                                                      test_name=
                                                                                      datasets[set_name]["filenames"][
                                                                                          2],
                                                                                      has_holdtarg=True)

    word_to_ix = TEXT.vocab.stoi
    ix_to_word = TEXT.vocab.itos

    VOCAB_SIZE = len(word_to_ix)

    word_embeds = TEXT.vocab.vectors

    model = MODEL(NUM_LABELS, VOCAB_SIZE,
                   EMBEDDING_DIM, HIDDEN_DIM, word_embeds,
                   NUM_POLARITIES, BATCH_SIZE, DROPOUT_RATE,
                   max_co_occurs=MAX_CO_OCCURS)

    print("num params = ")
    print(len(model.state_dict()))
    #    model.load_state_dict(torch.load("./model_states/adv_G_10.pt"))

    # Move the model to the GPU if available
    if using_GPU:
        model = model.cuda()

    train_c, dev_c, test_c, train_a, dev_a, test_a, losses, best_epoch = \
        train(train_data, dev_data, test_data,
              model,
              word_to_ix, ix_to_word,
              using_GPU)
    print("Train results: ")
    print("    " + str(train_c))
    print("    " + str(train_a))
    print("Dev results: ")
    print("    " + str(dev_c))
    print("    " + str(dev_a))
    print("Test results: ")
    print("    " + str(test_c))
    print("    " + str(test_a))
    print("Losses: ")
    print(losses)

    print("Best epoch = " + str(best_epoch))
    print("Train results: ")
    print("    " + str(train_c[best_epoch]) + " " + str(sum(train_c[best_epoch]) / len(train_c[best_epoch])))
    print("    " + str(train_a[best_epoch]))
    print("Dev results: ")
    print("    " + str(dev_c[best_epoch]) + " " + str(sum(dev_c[best_epoch]) / len(dev_c[best_epoch])))
    print("    " + str(dev_a[best_epoch]))
    print("Test results: ")
    print("    " + str(test_c[best_epoch]) + " " + str(sum(test_c[best_epoch]) / len(test_c[best_epoch])))
    print("    " + str(test_a[best_epoch]))

    print("saving model...")
    torch.save(model.state_dict(), "./model_states/adv_" + set_name + "_" + str(epochs) + ".pt")

    return model, TEXT, POLARITY


if __name__ == "__main__":
    main()