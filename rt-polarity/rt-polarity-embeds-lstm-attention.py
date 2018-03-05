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
import parser

epochs = 10
EMBEDDING_DIM = 50
HIDDEN_DIM = 2 * EMBEDDING_DIM
NUM_FEATURES = 6

class Model(nn.Module):
    def __init__(self, num_labels, vocab_size, embeddings_size,
                 hidden_dim, word_embeddings, num_features):
        super(Model, self).__init__()
        
        self.hidden_dim = hidden_dim

        self.word_embeds = nn.Embedding(vocab_size, embeddings_size)
        self.word_embeds.weight.data.copy_(torch.FloatTensor(word_embeddings))
        self.word_embeds.weight.requires_grad = False # don't update the embeddings

        self.feature_embeds = nn.Embedding(num_features, embeddings_size)
        
        # The LSTM takes [word embeddings, feature embeddings] as inputs, and
        # outputs hidden states with dimensionality hidden_dim.
        self.lstm = nn.LSTM(2 * embeddings_size, hidden_dim)

        # The linear layer that maps from hidden state space to target space
        self.hidden2label = nn.Linear(hidden_dim, num_labels)
        self.hidden = self.init_hidden()

        # Matrix of weights for each layer
        # Linear map from hidden layers to alpha for that layer
        self.attention = nn.Linear(hidden_dim, 1)

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        # Initialize to 0's
        # Returns (hidden state, cell state)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, word_vec, feature_vec): # how to classify a training example?
        # attention: learn weight matrix (parameter) mapping hidden layers to weights
        #       alpha_i = weight * hidden layer+i
        #       normalize alphas (through softmax)
        #       x_i = sum(a_{i,t} * x_t) (weighted sum)
        # sum of each hidden layer
        # attention: don't have to propogate information onto end

        # Features: word -> features (Tff file)
        # number from 0 - 5 (strong positive is 5, weak positive is 4, etc.)
        # learning embeddings for each of 0-5 (dimension: 50x5)
        # use to nn.Embeddings(5, embeddings_size)
        # lstm input is [word embeddings, feature embeddings]

        # Apply embeddings & prepare input
        # TODO: will ever need to deal with empty embeddings?
        word_embeds_vec = self.word_embeds(word_vec).view(len(word_vec), 1, -1)
        feature_embeds_vec = self.feature_embeds(feature_vec).view(len(feature_vec), 1, -1)
        # [word embeddings, feature embeddings]
        lstm_input = torch.cat((word_embeds_vec, feature_embeds_vec),
                               1).view(len(word_vec), 1, -1)

        # Pass through lstm
        lstm_out, self.hidden = self.lstm(lstm_input, self.hidden)

        # Compute and apply weights (attention) to each layer
        alphas = self.attention(lstm_out)
        alphas = F.softmax(alphas, dim=0)
        weighted_lstm_out = torch.sum(torch.mul(alphas, lstm_out), dim=0)
        
        # Get final results, passing in weighted lstm output:
        tag_space = self.hidden2label(weighted_lstm_out)
        #print("tags = " + str(tag_space))
        log_probs = F.log_softmax(tag_space, dim=1)
        return log_probs

# includes word and polarity ("feature") embeddings
def make_embeddings_vector(sentence, word_to_ix, word_to_polarity):
    word_vec = []
    feature_vec = []
    for word in sentence:
        # domains of word_to_ix and word_to_polarity are equal,
        # so just need to check word_to_ix
        if word in word_to_ix:
            word_vec.append(word_to_ix[word])
            feature_vec.append(word_to_polarity[word])
            #print(word + " " + str(word_to_polarity[word]))
    return autograd.Variable(torch.LongTensor(word_vec)), autograd.Variable(
        torch.LongTensor(feature_vec))

def train(Xpos, Xneg, model, word_to_ix, word_to_polarity, Xposdev, Xnegdev, Xpostest, Xnegtest):
    Xtrain = [Xs for Xs in Xpos]
    for Xs in Xneg:
        Xtrain.append(Xs)
    Xtrain = random.sample(Xtrain, len(Xtrain))

    print("Evaluating before training...")
    trains = []
    devs = []
    tests = []
    trainposperf, trainnegperf, trainperf = evaluate(model, word_to_ix, word_to_polarity, Xpos, Xneg)
    print("train correctly negative: " + str(trainnegperf))
    print("train correctly positive: " + str(trainposperf))
    print("train correctly correct: " + str(trainperf))
    devposperf, devnegperf, devperf = evaluate(model, word_to_ix, word_to_polarity, Xposdev, Xnegdev)
    print("dev correctly negative: " + str(devnegperf))
    print("dev correctly positive: " + str(devposperf))
    print("dev correctly correct: " + str(devperf))
    trains.append(trainperf)
    devs.append(devperf)
    tests.append(evaluate(model, word_to_ix, word_to_polarity, Xpostest, Xnegtest)[2])

    loss_function = nn.NLLLoss()

    # skip updating the non-requires-grad params (i.e. the embeddings)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    
    for epoch in range(0,epochs):
        print("Epoch " + str(epoch))
        i = 0
        for instance, label in Xtrain:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            model.hidden = model.init_hidden()

            # Step 2. Make our BOW vector and also we must wrap the target in a
            # Variable as an integer. For example, if the target is NEGATIVE, then
            # we wrap the integer 0. The loss function then knows that the 0th
            # element of the log probabilities is the log probability
            # corresponding to NEGATIVE
            words_vec, features_vec = make_embeddings_vector(instance, word_to_ix, word_to_polarity)
            target = autograd.Variable(torch.LongTensor([label])) #make_target(1, label_to_ix))

            # Step 3. Run our forward pass.
            log_probs = model(words_vec, features_vec)
            
            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss = loss_function(log_probs, target) # log_probs = actual distr, target = computed distr
            loss.backward()
            optimizer.step()
            #print("loss = " + str(loss))
            if (i % 100 == 0):
                print("    " + str(i))
            i += 1
        print("Evaluating...")
        trainposperf, trainnegperf, trainperf = evaluate(model, word_to_ix, word_to_polarity, Xpos, Xneg)
        print("train correctly negative: " + str(trainnegperf))
        print("train correctly positive: " + str(trainposperf))
        print("train correctly correct: " + str(trainperf))
        devposperf, devnegperf, devperf = evaluate(model, word_to_ix, word_to_polarity, Xposdev, Xnegdev)
        print("dev correctly negative: " + str(devnegperf))
        print("dev correctly positive: " + str(devposperf))
        print("dev correctly correct: " + str(devperf))
        trains.append(trainperf)
        devs.append(devperf)
        tests.append(evaluate(model, word_to_ix, word_to_polarity, Xpostest, Xnegtest)[2])
    return tests, devs, trains

def evaluate(model, word_to_ix, word_to_polarity, Xpostest, Xnegtest):
    count = 0
    # count positive classifications in pos
    for words, _ in Xpostest:
        w_vector, f_vector = make_embeddings_vector(words, word_to_ix, word_to_polarity)
        log_probs = model(w_vector, f_vector)
        if (log_probs.data[0][0] < log_probs.data[0][1]): # classify as positive
            count += 1
    pos = float(count) / float(len(Xpostest))
    
    # count negative classifications in neg
    negcount = 0
    for words, _ in Xnegtest:
        w_vector, f_vector = make_embeddings_vector(words, word_to_ix, word_to_polarity)
        log_probs = model(w_vector, f_vector)
        if (log_probs.data[0][0] > log_probs.data[0][1]): # classify as negative
            negcount += 1
    neg = float(negcount) / float(len(Xnegtest))
    
    total = float(count + negcount) / float(len(Xpostest) + len(Xnegtest))
    return pos, neg, total

# plot across each epoch
def plot(dev_accs, train_accs):
    plt.plot(range(0,epochs+1), dev_accs, c = 'red', label = 'Dev Set Accuracy')
    plt.plot(range(0,epochs+1), train_accs, c = 'blue', label = 'Train Set Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. # of Epochs')
    plt.legend()
    plt.show()

def main():
    Xneg = parser.parse_input_files("rt-polaritydata/rt-polarity.neg", 0)
    Xpos = parser.parse_input_files("rt-polaritydata/rt-polarity.pos", 1)

    Xnegtrain, Xnegdev, Xnegtest = parser.splitdata(Xneg)
    Xpostrain, Xposdev, Xpostest = parser.splitdata(Xpos)

    word_to_ix, embeds = parser.parse_embeddings("glove.6B." + str(EMBEDDING_DIM) + "d.txt")
    # returns mapping word->polarity for shared words in word->ix
    word_to_polarity = parser.parse_tff("subjclueslen1-HLTEMNLP05.tff", word_to_ix)

    NUM_LABELS = 2
    VOCAB_SIZE = len(word_to_ix)

    model = Model(NUM_LABELS, VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, embeds, NUM_FEATURES)
    
    test_c, dev_c, train_c = train(Xpostrain, Xnegtrain, model,
                                   word_to_ix, word_to_polarity,
                                   Xposdev, Xnegdev, Xpostest, Xnegtest)

    plot(dev_c, train_c)

    print(str(dev_c))
    best_epochs = np.argmax(np.array(dev_c))
    dev_results = dev_c[best_epochs]
    print("Test performance = " + str(test_c[best_epochs]))

if __name__ == "__main__":
    main()
