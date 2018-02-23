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

epochs = 10
EMBEDDING_DIM = 50
HIDDEN_DIM = 50

class Model(nn.Module):
    def __init__(self, num_labels, vocab_size, embeddings_size, hidden_dim, embeddings):
        super(Model, self).__init__()
        
        self.hidden_dim = hidden_dim

        self.embeds = nn.Embedding(vocab_size, embeddings_size)
        self.embeds.weight.data.copy_(torch.FloatTensor(embeddings))
        self.embeds.weight.requires_grad = False # don't update the embeddings

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embeddings_size, hidden_dim)

        # The linear layer that maps from hidden state space to target space
        self.hidden2tag = nn.Linear(hidden_dim, num_labels)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, word_vec): # how to classify a training example?
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
        
        #lin = self.linear(embeds)
        #mean = torch.mean(lin, dim=0).view(1, -1)
        #print(self.embeds(word_vec))
        embeds_vec = self.embeds(word_vec).view(len(word_vec), -1)
        #print(embeds_vec)
        lstm_out, self.hidden = self.lstm(embeds_vec, self.hidden)
        #print(str(lstm_out) + " " + str(lstm_out[-1]))
        # index "-1" equivalent to "len(lstm_out) -1", essentially getting the
        # result of the final time-step (after all words have been considered)
        tag_space = self.hidden2tag(lstm_out[-1])
        #print("tags = " + str(tag_space))
        log_probs = F.log_softmax(tag_space, dim=1)
        return log_probs

def make_embeddings_vector(sentence, word_to_ix):
    word_vec = []
    for word in sentence:
        if word in word_to_ix:
            word_vec.append(word_to_ix[word])
    return autograd.Variable(torch.LongTensor(word_vec))

def train(Xpos, Xneg, model, word_to_ix, Xposdev, Xnegdev, Xpostest, Xnegtest):
    loss_function = nn.NLLLoss()

    # skip updating the non-requires-grad params (i.e. the embeddings)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    
    Xtrain = [Xs for Xs in Xpos]
    for Xs in Xneg:
        Xtrain.append(Xs)
    Xtrain = random.sample(Xtrain, len(Xtrain))

    trains = []
    devs = []
    tests = []
    trainposperf, trainnegperf, trainperf = evaluate(model, word_to_ix, Xpos, Xneg)
    print("train correctly negative: " + str(trainnegperf))
    print("train correctly positive: " + str(trainposperf))
    print("train correctly correct: " + str(trainperf))
    devposperf, devnegperf, devperf = evaluate(model, word_to_ix, Xposdev, Xnegdev)
    print("dev correctly negative: " + str(devnegperf))
    print("dev correctly positive: " + str(devposperf))
    print("dev correctly correct: " + str(devperf))
    trains.append(trainperf)
    devs.append(devperf)
    tests.append(evaluate(model, word_to_ix, Xpostest, Xnegtest)[2])
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
            words_vec = make_embeddings_vector(instance, word_to_ix) # average embeddings
            target = autograd.Variable(torch.LongTensor([label])) #make_target(1, label_to_ix))

            # Step 3. Run our forward pass.
            log_probs = model(words_vec)
            
            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss = loss_function(log_probs, target) # log_probs = actual distr, target = computed distr
            loss.backward()
            optimizer.step()
            #print("loss = " + str(loss))
            if (i % 100 == 0):
                print("    " + str(i))
            i += 1
        trainposperf, trainnegperf, trainperf = evaluate(model, word_to_ix, Xpos, Xneg)
        print("train correctly negative: " + str(trainnegperf))
        print("train correctly positive: " + str(trainposperf))
        print("train correctly correct: " + str(trainperf))
        devposperf, devnegperf, devperf = evaluate(model, word_to_ix, Xposdev, Xnegdev)
        print("dev correctly negative: " + str(devnegperf))
        print("dev correctly positive: " + str(devposperf))
        print("dev correctly correct: " + str(devperf))
        trains.append(trainperf)
        devs.append(devperf)
        tests.append(evaluate(model, word_to_ix, Xpostest, Xnegtest)[2])
    return tests, devs, trains

def parse_embeddings(filename):
    word_to_ix = {} # index of word in embeddings
    embeds = []
    print("opening " + filename + "...")
    with open("glove.6B/" + filename, 'rt') as f:
        i = 0
        for line in f:
            word_and_embed = line.split()
            word = word_and_embed.pop(0)
            word_to_ix[word] = i
            embeds.append([float(val) for val in word_and_embed])
            if (i % 50000 == 49999): # 400,000 lines total
                print("parsed line " + str(i))
            i += 1
            """
            if (i > 100000):
                break
            """
    return word_to_ix, embeds

def splitdata(X):
    trainlen = int(0.8 * len(X))
    devlen = int(0.1 * len(X))
    Xtrain = []
    Xdev = []
    Xtest = []
    for i in range(0,len(X)):
        if (i < trainlen):
            Xtrain.append(X[i])
        elif (i < trainlen + devlen):
            Xdev.append(X[i])
        else:
            Xtest.append(X[i])
    print(len(Xtrain))
    print(len(Xdev))
    print(len(Xtest))

    return Xtrain, Xdev, Xtest

def parse_input_files(filename, label):
    """
    Reads the file with name filename
    """
    xs = []
    with open(filename, 'rt', encoding='latin1') as f:
        for line in f:
            xs.append((line.split(), label))
    return xs

def evaluate(model, word_to_ix, Xpostest, Xnegtest):
    count = 0
    # count positive classifications in pos
    for words, _ in Xpostest:
        w_vector = make_embeddings_vector(words, word_to_ix)
        log_probs = model(w_vector)
        if (log_probs.data[0][0] < log_probs.data[0][1]): # classify as positive
            count += 1
    pos = float(count) / float(len(Xpostest))
    
    # count negative classifications in neg
    negcount = 0
    for words, _ in Xnegtest:
        w_vector = make_embeddings_vector(words, word_to_ix)
        log_probs = model(w_vector)
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
    Xneg = parse_input_files("rt-polaritydata/rt-polarity.neg", 0)
    Xpos = parse_input_files("rt-polaritydata/rt-polarity.pos", 1)

    Xnegtrain, Xnegdev, Xnegtest = splitdata(Xneg)
    Xpostrain, Xposdev, Xpostest = splitdata(Xpos)

    word_to_ix, embeds = parse_embeddings("glove.6B." + str(EMBEDDING_DIM) + "d.txt")

    NUM_LABELS = 2
    VOCAB_SIZE = len(word_to_ix)

    model = Model(NUM_LABELS, VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, embeds)
    
    test_c, dev_c, train_c = train(Xpostrain, Xnegtrain, model, word_to_ix, Xposdev, Xnegdev, Xpostest, Xnegtest)

    plot(dev_c, train_c)

    print(str(dev_c))
    best_epochs = np.argmax(np.array(dev_c))
    dev_results = dev_c[best_epochs]
    print("Test performance = " + str(test_c[best_epochs]))

if __name__ == "__main__":
    main()
