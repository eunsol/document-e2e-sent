import numpy as np
import random
from random import shuffle
import torch
from torch import autograd
'''
from torchtext import data
from torchtext import datasets
'''
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

label_to_ix = {"NEGATIVE":0, "POSITIVE":1}

class Model(nn.Module):
    def __init__(self, num_labels, vocab_size):
        # calls the init function of nn.Module.  Dont get confused by syntax,
        # just always do it in an nn.Module
        super(Model, self).__init__()
        
        # Define the parameters that you will need.  In this case, we need A and b,
        # the parameters of the affine mapping.
        # Torch defines nn.Linear(), which provides the affine map.
        # Make sure you understand why the input dimension is vocab_size
        # and the output is num_labels!

        # linearly maps from index of word to number of labels (2)
        self.linear = nn.Linear(vocab_size, num_labels) # initialize self.linear
        # embedding size, num_labels

        # Don't need to define softmax because softmax doesn't take parameters

    def forward(self, word_vec): # how to classify a training example?
        # Pass the input through the linear layer,
        # then pass that through log_softmax (normalize to probabilities)
        # Many non-linearities and other functions are in torch.nn.functional
        return F.log_softmax(self.linear(word_vec), dim=1)

def main():
    '''
    #X = parse_input_files("rt-polaritydata/rt-polarity.neg", 0) # using 0 for classification task
    inputs = data.Field(lower=True)
    labels = data.Field(sequential=False) # don't apply tokenization
    train, dev, test = #load data (somehow)

    inputs.build_vocab(train, dev, test) #pre-existing word vectors?

    #build iterators (batching process)
    '''
    #do preprocessing simple way, then try running a nn,
    #then complexify preprocessing
    Xneg = parse_input_files("rt-polaritydata/rt-polarity.neg", 0)
    #Yneg = [0 for i in range(0,len(Xneg))]        
    Xpos = parse_input_files("rt-polaritydata/rt-polarity.pos", 1)
    #Ypos = [1 for i in range(0,len(Xpos))]

    Xnegtrain, Xnegdev, Xnegtest = splitdata(Xneg)
    Xpostrain, Xposdev, Xpostest = splitdata(Xpos)

    # word_to_ix maps each word in the vocab to a unique integer, which will be its
    # index into the Bag of words vector
    word_to_ix = {}
    for sent, _ in Xnegtrain + Xnegdev + Xnegtest + Xpostrain + Xposdev + Xpostest:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    #print(word_to_ix)
    
    VOCAB_SIZE = len(word_to_ix)
    NUM_LABELS = 2 # 0 or 1 for positive
    
    model = Model(NUM_LABELS, VOCAB_SIZE)
    
    # the model knows its parameters.  The first output below is A, the second is b.
    # Whenever you assign a component to a class variable in the __init__ function
    # of a module, which was done with the line
    # self.linear = nn.Linear(...)
    # Then through some Python magic from the Pytorch devs, your module
    # (in this case, BoWClassifier) will store knowledge of the nn.Linear's parameters
    for param in model.parameters():
        print(param) # prints A and b of the affline mapping

    evaluate(model, word_to_ix, Xpostest, Xnegtest)

    train(Xpostrain, Xnegtrain, model, word_to_ix)
    
    # To run the model, pass in a BoW vector, but wrapped in an autograd.Variable
#    bow_vector = make_bow_vector(Xnegtest[0][0], word_to_ix)
#    log_probs = model(autograd.Variable(bow_vector))
#    print(log_probs)

    correctness = evaluate(model, word_to_ix, Xposdev, Xnegdev)
    print(correctness)

def evaluate(model, word_to_ix, Xpostest, Xnegtest):
    count = 0
    # count positive classifications in pos
    for words, _ in Xpostest:
        log_probs_sum = autograd.Variable(torch.zeros(2).view(1,-1))
        for word in words:
            model.zero_grad()
            idx = word_to_ix[word]
            word_vec = autograd.Variable(torch.zeros(len(word_to_ix)))
            word_vec[idx] = 1
            word_vec_a = word_vec.view(1,-1)
            log_probs = model(word_vec_a)
            log_probs_sum += log_probs
        ave_prob = log_probs_sum / len(words) # get average
        #bow_vector = make_bow_vector(words, word_to_ix)
        #log_probs = model(autograd.Variable(bow_vector))
        if (log_probs_sum.data[0][0] < log_probs_sum.data[0][1]): # classify as positive
            count += 1
    print("correctly positive: " + str(float(count) / float(len(Xpostest))))
    
    # count negative classifications in neg
    negcount = 0
    for words, _ in Xnegtest:
        log_probs_sum = autograd.Variable(torch.zeros(2).view(1,-1))
        for word in words:
            model.zero_grad()
            idx = word_to_ix[word]
            word_vec = autograd.Variable(torch.zeros(len(word_to_ix)))
            word_vec[idx] = 1
            word_vec_a = word_vec.view(1,-1)
            log_probs = model(word_vec_a)
            log_probs_sum += log_probs
        ave_prob = log_probs_sum / len(words) # get average
        #bow_vector = make_bow_vector(words, word_to_ix)
        #log_probs = model(autograd.Variable(bow_vector))
        if (log_probs.data[0][0] > log_probs.data[0][1]): # classify as negative
            negcount += 1
    print("correctly negative: " + str(float(negcount) / float(len(Xnegtest))))
    
    print("correct: " + str(float(count + negcount) / float(len(Xpostest) + len(Xnegtest))))
    return (float(count + negcount) / float(len(Xpostest) + len(Xnegtest)))

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

def train(Xpos, Xneg, model, word_to_ix):
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Combine the + and - and randomize
    Xtrain = [Xs for Xs in Xpos]
    for Xs in Xneg:
        Xtrain.append(Xs)
    Xtrain = random.sample(Xtrain, len(Xtrain))

    # Usually you want to pass over the training data several times.
    # 100 is much bigger than on a real data set, but real datasets have more than
    # two instances.  Usually, somewhere between 5 and 30 epochs is reasonable.
    for epoch in range(1):
        print(epoch)
        i = 0
        print(len(Xtrain))
        for instance, label in Xtrain:
            print(label)
            model.zero_grad()

            model.
            '''
            log_probs_sum = autograd.Variable(torch.zeros(2).view(1,-1))
            for word in instance:
                model.zero_grad()
                idx = word_to_ix[word]
                word_vec = autograd.Variable(torch.zeros(len(word_to_ix)))
                word_vec[idx] = 1
                word_vec_a = word_vec.view(1,-1)
                log_probs = model(word_vec_a)
                log_probs_sum += log_probs
            ave_prob = log_probs_sum / len(instance) # get average
            '''
            label_ft = autograd.Variable(torch.LongTensor([label]))
            '''
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Make our BOW vector and also we must wrap the target in a
            # Variable as an integer. For example, if the target is NEGATIVE, then
            # we wrap the integer 0. The loss function then knows that the 0th
            # element of the log probabilities is the log probability
            # corresponding to NEGATIVE
            bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
            target = autograd.Variable(torch.LongTensor([label]))#make_target(1, label_to_ix))
            
            # Step 3. Run our forward pass.
            log_probs = model(bow_vec) # what our probability distr looks like currently
            '''
            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss = loss_function(ave_prob, label_ft)
            loss.backward()
            optimizer.step()
            i += 1
            if (i % 500 == 0):
                print(i)
#            evaluate(model, word_to_ix, Xpos, Xneg)


# counts frequency of each word in a sentence
def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        vec[word_to_ix[word]] += 1
    #print(vec) # vector w/ frequencies of each word
    print(str(vec))
    return vec.view(1, -1)

def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])

def parse_input_files(filename, label):
    """
    Reads the file with name filename
    """
    xs = []
    with open(filename, 'rt', encoding='latin1') as f:
        for line in f:
            xs.append((line.split(), label))
    return xs


if __name__ == "__main__":
    main()
