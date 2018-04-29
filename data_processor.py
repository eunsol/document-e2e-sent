from torchtext import data
from torchtext.data import Pipeline
import torch
import random
import spacy
import numpy as np

#spacy_en = spacy.load('en')

'''
def tokenizer(text):
    to_ret = [tok.text for tok in spacy_en.tokenizer(text)]
    return to_ret
'''

def dummy_tokenizer(text):
    # already in the form of a list
    return text


def custom_post(input, *args):
    if input == "<pad>":
        return [-1, -1]
    return [index if index != "<pad>" else [-1, -1] for index in input]


def parse_input_files(batch_size, embedding_dim, using_GPU, filepath="./data/new_annot/trainsplit_holdtarg",
                      train_name="train.json", dev_name="dev.json", test_name="test.json"):
    """
    Reads the file with name filename
    """
    print("creating fields")
    TEXT = data.Field(sequential=True, use_vocab=True, batch_first=True, tokenize=dummy_tokenizer, include_lengths=True)
    LABEL = data.Field(sequential=False, use_vocab=False, batch_first=True)
    POLARITY = data.Field(sequential=True, use_vocab=True, batch_first=True, tokenize=dummy_tokenizer)
    DOCID = data.Field(sequential=False, use_vocab=True, batch_first=True, tokenize=dummy_tokenizer)
    # may not need these two?
    # HOLDER = data.Field(sequential=True, use_vocab=True, batch_first=True, tokenize=tokenizer)
    # TARGET = data.Field(sequential=True, use_vocab=True, batch_first=True, tokenize=tokenizer)

    HOLDER_TARGET = data.Field(sequential=True, use_vocab=True, batch_first=True, tokenize=dummy_tokenizer)
    H_IND = data.Field(sequential=True, use_vocab=False, batch_first=True, postprocessing=Pipeline(custom_post),
                       include_lengths=True)
    T_IND = data.Field(sequential=True, use_vocab=False, batch_first=True, postprocessing=Pipeline(custom_post),
                       include_lengths=True)

    print("parsing data from file")
    train, val, test = data.TabularDataset.splits(
        path=filepath, train=train_name,
        validation=dev_name, test=test_name,
        format='json',
        fields={'token': ('text', TEXT), 'label': ('label', LABEL),
                #'holder': ('holder', HOLDER), 'target': ('target', TARGET),
                'polarity': ('polarity', POLARITY),
                'holder_target': ('holder_target', HOLDER_TARGET),
                'docid': ('docid', DOCID),
                'holder_index': ('holder_index', H_IND), 'target_index': ('target_index', T_IND)}
    )

    print("loading word embeddings")
    TEXT.build_vocab(train, val, test, vectors="glove.6B." + str(embedding_dim) + "d")
    POLARITY.build_vocab(train, val, test)
    print(POLARITY.vocab.stoi)
    HOLDER_TARGET.build_vocab(train, val, test)
    print(HOLDER_TARGET.vocab.stoi)
    DOCID.build_vocab(train, val, test)

    print("Train length = " + str(len(train.examples)))
    print("Dev length = " + str(len(val.examples)))
    print("Test length = " + str(len(test.examples)))
    #print(val.examples[0].text)
  
    device_usage = -1
    if using_GPU:
        device_usage = 0

    print("splitting & batching data")
    train_iter, val_iter, test_iter = data.Iterator.splits(
        (train, val, test), sort_key=lambda x: len(x.text),
        repeat=False,
        batch_sizes=(batch_size, len(val.examples), len(test.examples)),
        device=device_usage,
        sort_within_batch=True)

    print("Repeat = " + str(train_iter.repeat))

    return train_iter, val_iter, test_iter, TEXT, DOCID
