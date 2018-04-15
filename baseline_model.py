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
import matplotlib.pyplot as plt
from matplotlib import ticker
import data_processor as parser
from sklearn.metrics import f1_score

NUM_LABELS = 3
# convention: [NEG, NULL, POS]
epochs = 10
EMBEDDING_DIM = 50
HIDDEN_DIM = 3 * EMBEDDING_DIM
NUM_POLARITIES = 6
BATCH_SIZE = 10
DROPOUT_RATE = 0.2


# Fix issue with batch size 1 -- ? Idk what's wrong
# Batch first
# Masking the padding
# Dropout <Check>
# Sort batch by length
# Decaying learning rate over time
# Run on GPU

class Model(nn.Module):
    def __init__(self, num_labels, vocab_size, embeddings_size,
                 hidden_dim, word_embeddings, num_polarities, batch_size,
                 dropout_rate):
        super(Model, self).__init__()
        # self.dropout = nn.Dropout(dropout_rate)

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        # Specify embedding layers
        self.word_embeds = nn.Embedding(vocab_size, embeddings_size)
        self.word_embeds.weight.data.copy_(torch.FloatTensor(word_embeddings))
        self.word_embeds.weight.requires_grad = False  # don't update the embeddings
        self.feature_embeds = nn.Embedding(num_polarities + 1, embeddings_size)  # add 1 for <pad>
        self.holder_target_embeds = nn.Embedding(5, embeddings_size)  # add 2 for <pad> and <unk>

        # The LSTM takes [word embeddings, feature embeddings] as inputs, and
        # outputs hidden states with dimensionality hidden_dim.
        self.lstm = nn.LSTM(3 * embeddings_size, hidden_dim, dropout=dropout_rate, batch_first=True)

        # The linear layer that maps from hidden state space to target space
        self.hidden2label = nn.Linear(hidden_dim, num_labels)
        self.hidden = self.init_hidden()

        # Matrix of weights for each layer
        # Linear map from hidden layers to alpha for that layer
        self.attention = nn.Linear(hidden_dim, 1)

    def init_hidden(self):
        # The axes semantics are (minibatch_size, num_layers/seq_len, hidden_dim (!= embedding size))
        # Initialize to 0's
        # Returns (hidden state, cell state)
        return (autograd.Variable(torch.zeros(self.batch_size, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(self.batch_size, 1, self.hidden_dim)))

    def forward(self, word_vec, feature_vec, holder_target_vec):
        # Apply embeddings & prepare input
        word_embeds_vec = self.word_embeds(word_vec)
        feature_embeds_vec = self.feature_embeds(feature_vec)
        ht_embeds_vec = self.holder_target_embeds(holder_target_vec)
        #print(str(word_embeds_vec.size()) + " " + str(feature_embeds_vec.size()))

        # [word embeddings, feature embeddings]
        lstm_input = torch.cat((word_embeds_vec, feature_embeds_vec, ht_embeds_vec), 2)
        #print(lstm_input.size())

        # Pass through lstm
        lstm_out, self.hidden = self.lstm(lstm_input, self.hidden)

        # Compute and apply weights (attention) to each layer (so dim=1)
        alphas = self.attention(lstm_out)
        alphas = F.softmax(alphas, dim=1) # batch_size x num_layers x 1
        weighted_lstm_out = torch.sum(torch.mul(alphas, lstm_out), dim=1) # batch_size x hidden_dim
        '''
        weighted_lstm_out = lstm_out[-1]
        '''
        # Get final results, passing in weighted lstm output:
        tag_space = self.hidden2label(weighted_lstm_out) # batch_size x 1
        log_probs = F.log_softmax(tag_space, dim=1) # batch_size x 1
        return log_probs, alphas


# includes word and polarity ("feature") embeddings
def make_embeddings_vector(sentence, word_to_ix, word_to_polarity):
    print(sentence)
    word_vec = []
    feature_vec = []
    for word in sentence:
        # domains of word_to_ix and word_to_polarity are equal,
        # so just need to check word_to_ix
        if word in word_to_ix:
            word_vec.append(word_to_ix[word])
            feature_vec.append(word_to_polarity[word])
            # print(word + " " + str(word_to_polarity[word]))
    print(word_vec)
    return autograd.Variable(torch.LongTensor(word_vec)), autograd.Variable(
        torch.LongTensor(feature_vec))


def train(Xtrain, Xdev, Xtest,
          model, word_to_ix, ix_to_word):
    print("Evaluating before training...")
    train_res = []
    dev_res = []
    test_res = []

    '''
    # Just for 1 batch
    for batch in Xtrain:
        evaluate_sentence(model, word_to_ix, ix_to_word, batch)
        break
    '''
    print("evaluating training...")
    train_score = evaluate(model, word_to_ix, ix_to_word, Xtrain)
    print("train f1 scores = " + str(train_score))
    print(Xdev)
    dev_score = evaluate(model, word_to_ix, ix_to_word, Xdev)
    print("dev f1 scores = " + str(dev_score))
    train_res.append(train_score)
    dev_res.append(dev_score)
    test_res.append(evaluate(model, word_to_ix, ix_to_word, Xtest))

    loss_function = nn.NLLLoss()

    # skip updating the non-requires-grad params (i.e. the embeddings)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)

    for epoch in range(0, epochs):
        print("Epoch " + str(epoch))
        i = 0
        for batch in Xtrain:
            words, polarity, holder_target, label = batch.text, batch.polarity, batch.holder_target, batch.label

            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
            model.batch_size = len(label.data)  # set batch size

            model.hidden = model.init_hidden()  # detach from last instance

            # Step 3. Run our forward pass.
            log_probs, alphas = model(words, polarity, holder_target)

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss = loss_function(log_probs, label)  # log_probs = actual distr, target = computed distr
            # print("loss = " + str(loss.data))
            loss.backward()
            optimizer.step()
            # print("loss = " + str(loss))
            if (i % 100 == 0):
                print("    " + str(i))
            i += 1
        '''
        # Just for 1 batch
        for batch in Xtrain:
            evaluate_sentence(model, word_to_ix, ix_to_word, batch)
            break
        '''
        print("Evaluating...")
        train_score = evaluate(model, word_to_ix, ix_to_word, Xtrain)
        print("train f1 scores = " + str(train_score))
        print(Xdev)
        dev_score = evaluate(model, word_to_ix, ix_to_word, Xdev)
        print("dev f1 scores = " + str(dev_score))
        train_res.append(train_score)
        dev_res.append(dev_score)
        test_res.append(evaluate(model, word_to_ix, ix_to_word, Xtest))
    return train_res, dev_res, test_res


def evaluate_sentence(model, word_to_ix, ix_to_word, batch):
    words, polarity, holder_target, label = batch.text, batch.polarity, batch.holder_target, batch.label
    log_probs, attention = model(words, polarity, holder_target)
    input_sentence = decode(words[:, 0], ix_to_word)
    print(str(log_probs.data.max(1)[1]) + str(label))
    print(input_sentence)
    instance_attention = attention[:, 0].data.numpy()
    print(instance_attention)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(instance_attention, cmap='bone')
    fig.colorbar(cax)
    # Set up axes
    ax.set_yticklabels([''] + input_sentence)
    ax.set_xticklabels([''])

    # Show label at every tick
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def decode(word_indices, ix_to_word):
    words = [ix_to_word[index] for index in word_indices.data]
    return words


def evaluate(model, word_to_ix, ix_to_word, Xs):
    predictions = []
    truths = []

    print("Iterate across : " + str(len(Xs)) + " batch(es)")
    i = 0
    # count positive classifications in pos
    for batch in Xs:
        i += 1
        # print(word_to_ix)
        words, polarity, holder_target, label = batch.text, batch.polarity, batch.holder_target, batch.label
        model.batch_size = len(label.data)  # set batch size
        if len(label.data) > BATCH_SIZE:
            print(label.data)

        model.hidden = model.init_hidden()  # detaching it from its history on the last instance.
        log_probs, attention = model(words, polarity, holder_target) # log probs: batch_size x 3
        pred_label = log_probs.data.max(1)[1]

        predictions.extend(list(pred_label))
        truths.extend(list(label))
        if (i % 10 == 0):
            print(i)
        '''
        score = f1_score(list(pred_label), list(label.data.cpu().numpy()), labels=[-1, 0, 1], average=None)
        print(score)
        print
        '''
    print(list(pred_label))
    print(list(label.data.cpu().numpy()))
    score = f1_score(list(pred_label), list(label.data.cpu().numpy()), labels=[0, 1, 2], average=None)
    print(score)

    return score


# plot across each epoch
def plot(dev_accs, train_accs):
    plt.plot(range(0, epochs + 1), dev_accs, c='red', label='Dev Set Accuracy')
    plt.plot(range(0, epochs + 1), train_accs, c='blue', label='Train Set Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. # of Epochs')
    plt.legend()
    plt.show()


def main():
    train_data, dev_data, test_data, TEXT = parser.parse_input_files(BATCH_SIZE, EMBEDDING_DIM)

    word_to_ix = TEXT.vocab.stoi
    ix_to_word = TEXT.vocab.itos

    VOCAB_SIZE = len(word_to_ix)

    word_embeds = TEXT.vocab.vectors

    model = Model(NUM_LABELS, VOCAB_SIZE,
                  EMBEDDING_DIM, HIDDEN_DIM, word_embeds,
                  NUM_POLARITIES, BATCH_SIZE, DROPOUT_RATE)

    train_c, dev_c, test_c = train(train_data, dev_data, test_data,
                                   model,
                                   word_to_ix, ix_to_word)

    print(train_c)
    print(dev_c)
    print(test_c)
    '''                   
    plot(dev_c, train_c)

    print(str(dev_c))
    best_epochs = np.argmax(np.array(dev_c))
    dev_results = dev_c[best_epochs]
    print("Test performance = " + str(test_c[best_epochs]))
    '''


if __name__ == "__main__":
    main()
