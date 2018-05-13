import torch
import data_processor as parser
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor
import advanced_model_1
from advanced_model_1 import Model
import json
import argparse

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

set_name = "F"
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

BATCH_SIZE = 1

type_prior_polarity_to_polarity = {('strongsubj', 'negative'):"strongneg",
                                  ('weaksubj', 'negative'):"weakneg",
                                  ('weaksubj', 'neutral'):"neutral",
                                  ('strongsubj', 'neutral'):"neutral",
                                  ('weaksubj', 'positive'):"weakpos",
                                  ('strongsubj', 'positive'):"strongpos"
                                   }

word_to_polarity = {}
with open("./resources/subjclueslen1-HLTEMNLP05.tff", 'r') as f:
    i = 0
    for line in f:
        word_line = line.split()
        word = (word_line[2].split("="))[1]
        type_ = (word_line[0].split("="))[1]
        prior_polarity = (word_line[5].split("="))[1]
        # ignore "both"
        if prior_polarity == "both":
            continue
        polarity = type_prior_polarity_to_polarity[(type_, prior_polarity)]
        word_to_polarity[word] = polarity

def decode(word_indices, ix_to_word):
    words = [ix_to_word[index] for index in word_indices.data[0]]
    return words

def encode(word_indices, word_to_ix):
    indices = [word_to_ix[word] for word in word_indices.data[0]]
    return indices

def get_polarity(words, polarity_to_ix):
    polarities = [polarity_to_ix[word_to_polarity[word]] for word in words.data[0]]
    return polarities

def get_holders():


def main():
    model, TEXT = advanced_model_1.main()

    print()
    text = input("What is your text? ")
    holder_indices = input("What indices are your holders at? ")
    target_indices = input("What indices are your targets at? ")

    words = torch.LongTensor(encode(text, TEXT.vocab.stoi()))
    model.batch_size = 1

    model.eval()

    log_probs = model(words, polarity, lengths=len(text),
                    holders, targets, holder_lengths=len(holder_indices), target_lengths=len(target_indices),
                    co_occur_feature=co_occur_feature)  # log probs: batch_size x 3
    print(log_probs)

if __name__ == "__main__":
    main()
