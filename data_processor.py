from torchtext import data
import torch
import random
import spacy
import numpy as np

spacy_en = spacy.load('en')

# 0 is used to denote "unknown"/"empty"
type_priorpolarity_to_polarity = {('strongsubj', 'negative'): '1',
                                  ('weaksubj', 'negative'): '2',
                                  ('weaksubj', 'neutral'): '3',
                                  ('strongsubj', 'neutral'): '3',
                                  ('weaksubj', 'positive'): '4',
                                  ('strongsubj', 'positive'): '5'}


def parse_tff(filename, word_to_ix):
    # has identical words to word_to_ix
    word_to_polarity = {}
    if word_to_ix is not None:
        # initialize all words from word_to_ix to "unknown"
        for word in word_to_ix:
            word_to_polarity[word] = 0
    # read words which already have an associated polarity
    print("opening " + filename + "...")
    with open("./resources/" + filename, 'rt') as f:
        i = 0
        for line in f:
            word_line = line.split()
            word = (word_line[2].split("="))[1]
            if word_to_ix is not None:
                # don't include words not in word_to_ix
                if word not in word_to_ix:
                    continue
            type_ = (word_line[0].split("="))[1]
            priorpolarity = (word_line[5].split("="))[1]
            if priorpolarity == "both":
                continue
            polarity = type_priorpolarity_to_polarity[(type_, priorpolarity)]
            word_to_polarity[word] = polarity
            # print(word_to_polarity[word])
    return word_to_polarity


def parse_tff_ixs(filename, word_to_ix):
    # has identical words to word_to_ix
    ix_to_polarity = {}
    # initialize all words from word_to_ix to "unknown"
    for word in word_to_ix:
        ix_to_polarity[word_to_ix[word]] = 0
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
            ix_to_polarity[word_to_ix[word]] = polarity
            # print(word_to_polarity[word])
    return ix_to_polarity


def parse_embeddings(filename):
    word_to_ix = {}  # index of word in embeddings
    embeds = []
    print("opening " + filename + "...")
    with open("glove.6B/" + filename, 'rt') as f:
        i = 0
        for line in f:
            word_and_embed = line.split()
            word = word_and_embed.pop(0)
            word_to_ix[word] = i
            embeds.append([float(val) for val in word_and_embed])
            if (i % 50000 == 49999):  # 400,000 lines total
                print("parsed line " + str(i))
            i += 1
            """
            if (i > 100000):
                break
            """
    return word_to_ix, embeds


class MR(data.Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, polarity_field, label_field, path, pre_examples):
        """ Create an MR dataset instance given a path and fields.
        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            path: Path to the data file.
            examples: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        # self.fields = {'text':[('text',text_field)], 'label':[('label',label_field)]}
        # self.fields = [('text', text_field), ('polarity', polarity_field), ('label', label_field)]
        self.fields = [('text', text_field), ('polarity', polarity_field), ('label', label_field)]

        # path = self.dirname if path is None else path
        # print(self.fields.items())
        self.examples = []
        for pre_ex in pre_examples:
            example = data.Example.fromlist(pre_ex, self.fields)
            self.examples.append(example)
            # print(str(example.text) + " " + str(example.polarity))
        print(len(self.examples))
        super(MR, self).__init__(self.examples, self.fields)

    def get_examples():
        return self.examples


# does data preprocessing whereas other doesn't
def splitdata_batches(X, batch_size, embedding_dim):
    train, dev, test = splitdata(X, batch_size)
    print('setting up data')
    # '''
    TEXT = data.Field(sequential=True, tokenize=tokenizer)  # , lower=True)
    # lower means to lowercase
    POLARITY = data.Field(sequential=True, tokenize=get_polarity)
    LABEL = data.Field(sequential=False, use_vocab=False)

    # load data
    print('loading data')
    Xtrain = MR(TEXT, POLARITY, LABEL, None, train)
    Xdev = MR(TEXT, POLARITY, LABEL, None, dev)
    Xtest = MR(TEXT, POLARITY, LABEL, None, test)
    # print(Xtrain.fields.items())

    # load embeddings
    TEXT.build_vocab(Xtrain)
    POLARITY.build_vocab(Xtrain)  # should just be 0-5

    print(POLARITY.vocab.stoi)

    print('loading glove embeddings')
    TEXT.vocab.load_vectors('glove.6B.' + str(embedding_dim) + 'd')

    print('building batches')
    Xtrain, Xdev, Xtest = data.Iterator.splits(
        (Xtrain, Xdev, Xtest),
        batch_sizes=(batch_size, len(Xdev), len(Xtest)),
        repeat=False,
        device=-1  # run on CPU
    )

    return Xtrain, Xdev, Xtest, TEXT


def splitdata(X, batch_size):
    print('splitting data')

    trainlen = int(0.4 * len(X))
    # ensure all batches are the same size
    while trainlen % batch_size != 0:
        trainlen -= 1

    devlen = int(0.25 * (len(X) - 2 * trainlen))
    Xtrain = []
    Xdev = []
    Xtest = []
    for i in range(0, int(len(X) / 2)):
        if (i < trainlen):
            Xtrain.append(X[i])
            Xtrain.append(X[len(X) - 1 - i])
        elif (i < trainlen + devlen):
            Xdev.append(X[i])
            Xdev.append(X[len(X) - 1 - i])
        else:
            Xtest.append(X[i])
            Xtest.append(X[len(X) - 1 - i])

    # shuffle train data
    Xtrain = random.sample(Xtrain, len(Xtrain))

    print(len(Xtrain))
    print(len(Xdev))
    print(len(Xtest))
    return Xtrain, Xdev, Xtest


def tokenizer(text):
    to_ret = [tok.text for tok in spacy_en.tokenizer(text)]
    # print(to_ret)
    return to_ret


word_to_polarity = parse_tff("subjclueslen1-HLTEMNLP05.tff", None)


def get_polarity(text):
    tokens = text  # tokenizer(text) -- text is pre-tokenized
    to_ret = []
    for text in tokens:
        if text in word_to_polarity:
            to_ret.append(word_to_polarity[text])
        else:  # unknown
            to_ret.append('<unk>')
        if not (isinstance(to_ret[len(to_ret) - 1], str)):
            print(to_ret[len(to_ret) - 1])
    print(to_ret)
    return to_ret


def dummy_tokenizer(text):
    # already in the form of a list
    return text


def parse_input_files(batch_size, embedding_dim):
    """
    Reads the file with name filename
    """
    print("creating fields")
    TEXT = data.Field(sequential=True, use_vocab=True, batch_first=True, tokenize=dummy_tokenizer)
    LABEL = data.Field(sequential=False, use_vocab=False, batch_first=True)
    POLARITY = data.Field(sequential=True, use_vocab=True, batch_first=True, tokenize=dummy_tokenizer)
    # may not need these two?
    HOLDER = data.Field(sequential=True, use_vocab=True, batch_first=True,
                        tokenize=tokenizer)
    TARGET = data.Field(sequential=True, use_vocab=True, batch_first=True,
                        tokenize=tokenizer)

    HOLDER_TARGET_ANNOT = data.Field(sequential=True, use_vocab=False, batch_first=True)
    H_IND = data.Field(sequential=False, use_vocab=False, batch_first=True)
    T_IND = data.Field(sequential=False, use_vocab=False, batch_first=True)

    print("parsing data from file")
    train, val, test = data.TabularDataset.splits(
        path='./data/new_annot/polarity_label', train='new_train.json',
        validation='acl_dev_eval_new.json', test='acl_test_new.json',
        format='json',
        fields={'token': ('text', TEXT), 'label': ('label', LABEL),
                #'holder': ('holder', HOLDER), 'target': ('target', TARGET),
                'polarity': ('polarity', POLARITY)}
                #'holder_target': ('holder_target_annot', HOLDER_TARGET_ANNOT)}
                #'holder_index': ('h_ind', H_IND), 'target_index': ('t_ind', T_IND)}
    )

    print("loading word embeddings")
    TEXT.build_vocab(train, val, test, vectors="glove.6B." + str(embedding_dim) + "d")
    print(TEXT.vocab.stoi)
    POLARITY.build_vocab(train, val, test)
    print(POLARITY.vocab.stoi)

    print("Train length = " + str(len(train.examples)))
    print("Dev length = " + str(len(val.examples)))
    print("Test length = " + str(len(test.examples)))
    print(val.examples[0].text)

    print("splitting & batching data")
    train_iter, val_iter, test_iter = data.Iterator.splits(
        (train, val, test), sort_key=lambda x: len(x.text),
        repeat=False,
        batch_sizes=(batch_size, len(val.examples), len(test.examples)), device=-1)

    print("Repeat = " + str(train_iter.repeat))

    return train_iter, val_iter, test_iter, TEXT
