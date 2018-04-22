from torchtext import data
import torch
import random
import spacy
import numpy as np

spacy_en = spacy.load('en')


def tokenizer(text):
    to_ret = [tok.text for tok in spacy_en.tokenizer(text)]
    return to_ret


def dummy_tokenizer(text):
    # already in the form of a list
    return text


def parse_input_files(batch_size, embedding_dim, using_GPU, filepath="./data/new_annot/trainsplit_holdtarg",
                      train_name="train.json", dev_name="dev.json", test_name="test.json"):
    """
    Reads the file with name filename
    """
    print("creating fields")
    TEXT = data.Field(sequential=True, use_vocab=True, batch_first=True, tokenize=dummy_tokenizer, include_lengths=True)
    LABEL = data.Field(sequential=False, use_vocab=False, batch_first=True)
    POLARITY = data.Field(sequential=True, use_vocab=True, batch_first=True, tokenize=dummy_tokenizer)
    # may not need these two?
    HOLDER = data.Field(sequential=True, use_vocab=True, batch_first=True,
                        tokenize=tokenizer)
    TARGET = data.Field(sequential=True, use_vocab=True, batch_first=True,
                        tokenize=tokenizer)

    HOLDER_TARGET = data.Field(sequential=True, use_vocab=True, batch_first=True, tokenize=dummy_tokenizer)
    H_IND = data.Field(sequential=False, use_vocab=False, batch_first=True)
    T_IND = data.Field(sequential=False, use_vocab=False, batch_first=True)

    print("parsing data from file")
    train, val, test = data.TabularDataset.splits(
        path=filepath, train=train_name,
        validation=dev_name, test=test_name,
        format='json',
        fields={'token': ('text', TEXT), 'label': ('label', LABEL),
                #'holder': ('holder', HOLDER), 'target': ('target', TARGET),
                'polarity': ('polarity', POLARITY),
                'holder_target': ('holder_target', HOLDER_TARGET)}
                #'holder_index': ('h_ind', H_IND), 'target_index': ('t_ind', T_IND)}
    )

    print("loading word embeddings")
    TEXT.build_vocab(train, val, test, vectors="glove.6B." + str(embedding_dim) + "d")
    POLARITY.build_vocab(train, val, test)
    print(POLARITY.vocab.stoi)
    HOLDER_TARGET.build_vocab(train, val, test)
    print(HOLDER_TARGET.vocab.stoi)

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

    return train_iter, val_iter, test_iter, TEXT
