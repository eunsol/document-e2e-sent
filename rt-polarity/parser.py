# 0 is used to denote "unknown"/"empty"
type_priorpolarity_to_polarity = {('strongsubj', 'negative'):1,
                                  ('weaksubj', 'negative'):2,
                                  ('weaksubj', 'neutral'):3,
                                  ('strongsubj', 'neutral'):3,
                                  ('weaksubj', 'positive'):4,
                                  ('strongsubj', 'positive'):5 }

def parse_tff(filename, word_to_ix):
    # has identical words to word_to_ix
    word_to_polarity = {}
    # initialize all words from word_to_ix to "unknown"
    for word in word_to_ix:
        word_to_polarity[word] = 0
    # read words which already have an associated polarity
    print("opening " + filename + "...")
    with open("../resources/" + filename, 'rt') as f:
        i = 0
        for line in f:
            word_line = line.split()
            word = (word_line[2].split("="))[1]
            # don't include words not in word_to_ix
            if word not in word_to_ix:
                continue
            type_ = (word_line[0].split("="))[1]
            priorpolarity = (word_line[5].split("="))[1]
            if priorpolarity == "both":
                continue
            polarity = type_priorpolarity_to_polarity[(type_, priorpolarity)]
            word_to_polarity[word] = polarity
    return word_to_polarity

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
