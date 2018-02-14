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

class Model(nn.Module):
    def __init__(self, num_labels, vocab_size, embeddings_size):
        super(Model, self).__init__()
        
        self.linear = nn.Linear(embeddings_size, num_labels) # initialize self.linear

    def forward(self, embeds): # how to classify a training example?
        #embeds = self.embeddings(inputs).view((1, -1)) # get embeddings of inputs
        #print(str(self.embeddings(inputs)))
        # Pass the input through the linear layer,
        # then pass that through log_softmax (normalize to probabilities)
        # Many non-linearities and other functions are in torch.nn.functional
        return F.log_softmax(self.linear(embeds), dim=1)

def make_embeddings_vector(sentence, word_to_ix, embeds):
    vec = torch.zeros(len(embeds[0])) # width of embeds
    num_words = 0
    for word in sentence:
        if word in word_to_ix.keys():
            num_words += 1
            embeds_w = embeds[word_to_ix[word]]
            vec += torch.FloatTensor(embeds_w)
    if (num_words != 0):
        vec = vec / num_words
    return vec.view(1, -1)

def train(Xpos, Xneg, model, word_to_ix, embeds, Xposdev, Xnegdev, Xpostest, Xnegtest):
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    Xtrain = [Xs for Xs in Xpos]
    for Xs in Xneg:
        Xtrain.append(Xs)
    Xtrain = random.sample(Xtrain, len(Xtrain))

    evaluations = []
    tests = []
    for epoch in range(10):
        print(epoch)
        for instance, label in Xtrain:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Make our BOW vector and also we must wrap the target in a
            # Variable as an integer. For example, if the target is NEGATIVE, then
            # we wrap the integer 0. The loss function then knows that the 0th
            # element of the log probabilities is the log probability
            # corresponding to NEGATIVE
#            bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
#            context_idxs = [word_to_ix[word] for word in instance] # get the index of each word in the sentence
            embeds_vec = make_embeddings_vector(instance, word_to_ix, embeds) # average embeddings
            target = autograd.Variable(torch.LongTensor([label])) #make_target(1, label_to_ix))

            # Step 3. Run our forward pass.
#            log_probs = model(bow_vec) # what our probability distr looks like currently
            log_probs = model(autograd.Variable(embeds_vec))
            
            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss = loss_function(log_probs, target) # log_probs = actual distr, target = computed distr
            loss.backward()
            optimizer.step()
            #print("loss = " + str(loss))
        trainposperf, trainnegperf, trainperf = evaluate(model, word_to_ix, Xpos, Xneg)
        print("train correctly negative: " + str(trainnegperf))
        print("train correctly positive: " + str(trainposperf))
        print("train correctly correct: " + str(trainperf))
        devposperf, devnegperf, devperf = evaluate(model, word_to_ix, Xposdev, Xnegdev)
        print("dev correctly negative: " + str(devnegperf))
        print("dev correctly positive: " + str(devposperf))
        print("dev correctly correct: " + str(devperf))
        evaluations.append(devperf)
        tests.append(evaluate(model, word_to_ix, Xpostest, Xnegtest)[2])
    return tests, evaluations

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
        emb_vector = make_embeddings_vector(words, word_to_ix, embeds)
        log_probs = model(autograd.Variable(emb_vector))
        if (log_probs.data[0][0] < log_probs.data[0][1]): # classify as positive
            count += 1
    pos = float(count) / float(len(Xpostest))
    
    # count negative classifications in neg
    negcount = 0
    for words, _ in Xnegtest:
        emb_vector = make_embeddings_vector(words, word_to_ix, embeds)
        log_probs = model(autograd.Variable(emb_vector))
        if (log_probs.data[0][0] > log_probs.data[0][1]): # classify as negative
            negcount += 1
    neg = float(negcount) / float(len(Xnegtest))
    
    total = float(count + negcount) / float(len(Xpostest) + len(Xnegtest))
    return pos, neg, total

Xneg = parse_input_files("rt-polaritydata/rt-polarity.neg", 0)
Xpos = parse_input_files("rt-polaritydata/rt-polarity.pos", 1)

Xnegtrain, Xnegdev, Xnegtest = splitdata(Xneg)
Xpostrain, Xposdev, Xpostest = splitdata(Xpos)

word_to_ix, embeds = parse_embeddings("glove.6B.50d.txt")
embeddings = nn.Embedding(len(word_to_ix), len(embeds[0])) # 50 features per word
embeddings.weight.data.copy_(torch.FloatTensor(embeds)) # set the weights
                                               # to the pre-trained vector
'''
word_to_ix = {}
for sent, _ in Xnegtrain + Xnegdev + Xnegtest + Xpostrain + Xposdev + Xpostest:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
'''

NUM_LABELS = 2
VOCAB_SIZE = len(word_to_ix)
EMBEDDING_SIZE = 50

model = Model(NUM_LABELS, VOCAB_SIZE, EMBEDDING_SIZE)
pos,neg,total = evaluate(model, word_to_ix, Xpostrain, Xnegtrain)
print("dev correct: " + str(total))
test_c, dev_c = train(Xpostrain, Xnegtrain, model, word_to_ix, embeds, Xposdev, Xnegdev, Xpostest, Xnegtest)
print(str(dev_c))
best_epochs = np.argmax(np.array(dev_c))
dev_results = dev_c[best_epochs]
print("Test performance = " + str(test_c[best_epochs]))
