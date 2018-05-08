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
from allennlp.modules import FeedForward
from allennlp.nn import util
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor

NUM_LABELS = 3
# convention: [NEG, NULL, POS]
epochs = 10
EMBEDDING_DIM = 50
MAX_CO_OCCURS = 10
HIDDEN_DIM = EMBEDDING_DIM
NUM_POLARITIES = 6
DROPOUT_RATE = 0.2
using_GPU = torch.cuda.is_available()
threshold = torch.log(torch.FloatTensor([0.5, 0.2, 0.5]))
if using_GPU:
    threshold = threshold.cuda()

set_name = "E"
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

def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


class Model(nn.Module):
    def __init__(self, num_labels, vocab_size, embeddings_size,
                 hidden_dim, word_embeddings, num_polarities, batch_size,
                 dropout_rate, max_co_occurs):
        super(Model, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        # Specify embedding layers
        self.word_embeds = nn.Embedding(vocab_size, embeddings_size)
        self.word_embeds.weight.data.copy_(torch.FloatTensor(word_embeddings))
        self.word_embeds.weight.requires_grad = False  # don't update the embeddings
        self.polarity_embeds = nn.Embedding(num_polarities + 1, embeddings_size)  # add 1 for <pad>
        self.co_occur_embeds = nn.Embedding(max_co_occurs, embeddings_size)

        # The LSTM takes [word embeddings, feature embeddings] as inputs, and
        # outputs hidden states with dimensionality hidden_dim.
        self.lstm = nn.LSTM(2 * embeddings_size, hidden_dim, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=dropout_rate)

        # The linear layer that maps from hidden state space to target space
        self.hidden2label = nn.Linear(2 * hidden_dim, num_labels)

        # Matrix of weights for each layer
        # Linear map from hidden layers to alpha for that layer
        #self.attention = nn.Linear(2 * hidden_dim, 1)
        # Attempting feedforward attention, using 2 layers and sigmoid activation fxn
        # Last layer acts as the w_alpha layer
        self.attention = FeedForward(input_dim=2 * hidden_dim, num_layers=2, hidden_dims=[hidden_dim, 1],
                                     activations=nn.Sigmoid())

        # Span embeddings
        self._endpoint_span_extractor = EndpointSpanExtractor(2 * hidden_dim,
                                                              combination="x,y")
        self._attentive_span_extractor = SelfAttentiveSpanExtractor(input_dim=2 * hidden_dim)

        '''
        # FFNN for holder/target spans respectively
        self.holder_FFNN = FeedForward(input_dim=3*2*hidden_dim, num_layers=2, hidden_dims=[hidden_dim, hidden_dim],
                                       activations=nn.ReLU())
        self.target_FFNN = FeedForward(input_dim=3*2*hidden_dim, num_layers=2, hidden_dims=[hidden_dim, hidden_dim],
                                       activations=nn.ReLU())
        '''

        # Scoring pairwise sentiment: linear score approach
        self.pairwise_sentiment_score = FeedForward(input_dim=13 * hidden_dim, num_layers=2,
                                                    hidden_dims=[hidden_dim, num_labels],
                                                    activations=nn.ReLU())

    def forward(self, word_vec, feature_vec, lengths=None,
                holder_inds=None, target_inds=None, holder_lengths=None, target_lengths=None,
                co_occur_feature=None):
        # Apply embeddings & prepare input
        word_embeds_vec = self.word_embeds(word_vec)
        feature_embeds_vec = self.polarity_embeds(feature_vec)
        # print(str(word_embeds_vec.size()) + " " + str(feature_embeds_vec.size()))

        # [word embeddings, feature embeddings]
        lstm_input = torch.cat((word_embeds_vec, feature_embeds_vec), 2)
        # print(lstm_input.size())
        #        total_length = lstm_input.size(1)  # get the max sequence length

        # Mask out padding
        if lengths is not None:
            lengths = lengths.view(-1).tolist()
            # print(lengths)
            lstm_input = pack_padded_sequence(lstm_input, lengths, batch_first=True)
            # print(lstm_input.data.size())

        # Pass through lstm, encoding words
        lstm_out, _ = self.lstm(lstm_input)

        # Re-apply padding
        if lengths is not None:
            lstm_out = pad_packed_sequence(lstm_out, batch_first=True)[0]

        # Dimension: batch_size, sequence_len, 2 * hidden_dim (since concatenated forward w/ backward)
        lstm_out = self.dropout(lstm_out)

        # Mask encoded words to isolate holders
        holder_mask = (holder_inds[:, :, 0] >= 0).long()
        # holders = concatenated [start token, end token] across hidden dimension
        # Dimension: batch_size, # of holder mentions, 2 * (2 * hidden_dim)--both endpoints
        endpoint_holder = self._endpoint_span_extractor(lstm_out, holder_inds, span_indices_mask=holder_mask)
        # Dimension: batch_size, # of holder mentions, 2 * hidden_dim
        attended_holder = self._attentive_span_extractor(lstm_out, holder_inds, span_indices_mask=holder_mask)
        # Shape: (batch_size, # of holder mentions, 3 * (2 * hidden_dim))
        holders = torch.cat([endpoint_holder, attended_holder], -1)

        # Mask encoded words to isolate targets
        target_mask = (target_inds[:, :, 0] >= 0).long()
        # targets = concatenated [start token, end token] across hidden dimension
        # Dimension: batch_size, # of target mentions, 2 * (2 * hidden_dim)
        endpoint_target = self._endpoint_span_extractor(lstm_out, target_inds, span_indices_mask=target_mask)
        # Dimension: batch_size, # of target mentions, 2 * hidden_dim
        attended_target = self._attentive_span_extractor(lstm_out, target_inds, span_indices_mask=target_mask)
        # Shape: (batch_size, # of target mentions, 3 * (2 * hidden_dim))
        targets = torch.cat([endpoint_target, attended_target], -1)

        # Compute representation for each holder and target span, taking the mean across mentions
        # Shape: (batch_size, 3 * (2 * hidden_dim))
        holder_reps = torch.squeeze(torch.mean(holders, dim=1))
        target_reps = torch.squeeze(torch.mean(targets, dim=1))
        '''
        holder_alphas = self.holder_attention(holders)
        holder_alphas = F.softmax(holder_alphas, dim=1)  # batch_size x seq_len x 1
        # Sum across all holder spans
        weighted_holder_reps = torch.sum(torch.mul(alphas, lstm_out), dim=1)  # batch_size x hidden_dim
        target_alphas = self.holder_attention(holders)
        target_alphas = F.softmax(holder_alphas, dim=1)  # batch_size x seq_len x 1
        weighted_target_reps = torch.sum(torch.mul(alphas, lstm_out), dim=1)  # batch_size x hidden_dim
        '''

        # Apply embeds for co-occur feature
        # Shape: (batch_size, hidden_dim)
        co_occur_feature[co_occur_feature >= 10] = 9
        co_occur_embeds_vec = self.co_occur_embeds(co_occur_feature)

        # Get final pairwise score, passing in holder and target representations
        # Shape: (batch_size, hidden_dim + 2 * (3 * (2 * hidden_dim)))
        final_rep = torch.cat([holder_reps, target_reps, co_occur_embeds_vec], dim=-1)

        # print(co_occur_embeds_vec)
        output = self.pairwise_sentiment_score(final_rep)
        '''
        dimension = 1
        # Compute and apply weights (attention) to each layer (so dim=1)
        alphas = self.attention(lstm_out)
        alphas = F.softmax(alphas, dim=dimension)  # batch_size x num_layers x 1
        weighted_lstm_out = torch.sum(torch.mul(alphas, lstm_out), dim=dimension)  # batch_size x hidden_dim
        # weighted_lstm_out = lstm_out[-1]
        # Get final results, passing in weighted lstm output:
        output = self.hidden2label(weighted_lstm_out)  # batch_size x 1
        '''
        log_probs = F.log_softmax(output, dim=1)  # batch_size x 1

        return log_probs  #, alphas


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
            (words, lengths), polarity, label = batch.text, batch.polarity, batch.label
            (holders, holder_lengths) = batch.holder_index
            (targets, target_lengths) = batch.target_index
            co_occur_feature = batch.co_occurrences

            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
            model.batch_size = len(label.data)  # set batch size

            # Step 3. Run our forward pass.
            log_probs = model(words, polarity, lengths,
                              holders, targets, holder_lengths, target_lengths,
                              co_occur_feature=co_occur_feature)  # log probs: batch_size x 3

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
        if (epoch % 10 == 0):
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
        if (dev_f1_aves[epoch] > dev_f1_aves[best_epoch]):
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
        (words, lengths), polarity, label = batch.text, batch.polarity, batch.label
        (holders, holder_lengths) = batch.holder_index
        (targets, target_lengths) = batch.target_index
        co_occur_feature = batch.co_occurrences

        #words.no_grad() = lengths.no_grad() = polarity.no_grad() = label.no_grad() = True
        #holders.no_grad() = targets.no_grad() = holder_lengths.no_grad() = target_lengths.no_grad() = True

        model.batch_size = len(label.data)  # set batch size
        '''
        if len(label.data) > BATCH_SIZE:
            print(label.data)
        '''
        log_probs = model(words, polarity, lengths,
                          holders, targets, holder_lengths, target_lengths,
                          co_occur_feature=co_occur_feature)  # log probs: batch_size x 3
        if losses is not None:
            loss = loss_fxn(log_probs, label)
            loss_this_batch.append(float(loss))

        pred_label = log_probs.data.max(1)[1]  # torch.ones(len(log_probs), dtype=torch.long)
        '''
        if using_GPU:
            pred_label = pred_label.cuda()
        print(log_probs[:, 0] > threshold[0])
        # predict 0 if index at 0 > 0.5
        pred_label[log_probs[:, 0] > threshold[0]] = 2
        # predict 2 if index at 2 > 0.5
        pred_label[log_probs[:, 2] > threshold[2]] = 0
        pred_label = pred_label.data
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
    train_data, dev_data, test_data, TEXT, DOCID = parser.parse_input_files(BATCH_SIZE, EMBEDDING_DIM, using_GPU,
                                                                            filepath=datasets[set_name]["filepath"],
                                                                            train_name=datasets[set_name]["filenames"][0],
                                                                            dev_name=datasets[set_name]["filenames"][1],
                                                                            test_name=datasets[set_name]["filenames"][2],
                                                                            has_holdtarg=True)

    word_to_ix = TEXT.vocab.stoi
    ix_to_word = TEXT.vocab.itos

    VOCAB_SIZE = len(word_to_ix)

    word_embeds = TEXT.vocab.vectors

    model = Model(NUM_LABELS, VOCAB_SIZE,
                  EMBEDDING_DIM, HIDDEN_DIM, word_embeds,
                  NUM_POLARITIES, BATCH_SIZE, DROPOUT_RATE,
                  max_co_occurs=MAX_CO_OCCURS)

    print("num params = ")
    print(len(model.state_dict()))
    model.load_state_dict(torch.load("./model_states/adv_E_10.pt"))

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
    torch.save(model.state_dict(), "./model_states/adv_" + set_name + "_" + str(epochs + 10) + ".pt")
 
    '''                   
    print(str(dev_c))
    best_epochs = np.argmax(np.array(dev_c))
    dev_results = dev_c[best_epochs]
    print("Test performance = " + str(test_c[best_epochs]))
    '''


if __name__ == "__main__":
    main()
