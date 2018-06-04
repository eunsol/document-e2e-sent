import torch
import data_processor as parser
from advanced_model_1 import Model1
from baseline_model_GPU import Model
from train_and_evaluate import evaluate

#  from advanced_model_2 import Model2

NUM_LABELS = 3
# convention: [NEG, NULL, POS]
EMBEDDING_DIM = 50
MAX_CO_OCCURS = 10
MAX_NUM_MENTIONS = 10
HIDDEN_DIM = EMBEDDING_DIM
NUM_POLARITIES = 6
DROPOUT_RATE = 0.2
using_GPU = torch.cuda.is_available()
MODEL = Model1

epochs = [6, 9]
set_name = "C"
ablations_to_use = ["sentence", "co_occurrence", "num_mentions", "mentions_rank", "all"]
ABLATIONS = None  # ablations_to_use[4]


def save_name(epoch):
    if MODEL == Model:
        return "./model_states/baseline_" + set_name + "_" + str(epoch) + ".pt"
    if ABLATIONS is not None:
        return "./model_states/final/" + set_name + "/" + ABLATIONS + "/adv_" + str(epoch) + ".pt"
    else:
        return "./model_states/final/" + set_name + "/adv_" + str(epoch) + ".pt"


print(save_name("<epoch>"))

datasets = {"A": {"filepath": "./data/new_annot/feature",
                  "filenames": ["new_train.json", "acl_dev_eval_new.json", "acl_test_new.json"],
                  "weights": torch.FloatTensor([0.8, 1.825, 1]),
                  "batch": 10},
            "B": {"filepath": "./data/new_annot/trainsplit_holdtarg",
                  "filenames": ["train.json", "dev.json", "test.json"],
                  "weights": torch.FloatTensor([0.77, 1.766, 1]),
                  "batch": 10},
            "C": {"filepath": "./data/final",
                  "filenames": ["C_train.json", "acl_dev_eval.json", "acl_test.json", "acl_mpqa_eval.json"],
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
            "F": {"filepath": "./data/final",
                  "filenames": ["F_train.json", "acl_dev_eval.json", "acl_test.json", "acl_mpqa_eval.json"],
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
                  "batch": 50},
            "has_co_occurs": {"filepath": "./data/has_co_occurs",
                              "filenames": ["F_train.json", "acl_dev_eval_new.json", "acl_test_new.json"],
                              "weights": torch.FloatTensor([1.06656, 0.13078, 1]),
                              "batch": 80},
            "no_co_occurs": {"filepath": "./data/no_co_occurs",
                             "filenames": ["F_train.json", "acl_dev_eval_new.json", "acl_test_new.json"],
                             "weights": torch.FloatTensor([1, 0.02429, 1.206897]),
                             "batch": 80}
            }

BATCH_SIZE = datasets[set_name]["batch"]


def main():
    Xtrain, Xdev, ACLtest, TEXT, DOCID, POLARITY = parser.parse_input_files(BATCH_SIZE, EMBEDDING_DIM, using_GPU,
                                                                          filepath=datasets[set_name]["filepath"],
                                                                          train_name=datasets[set_name]["filenames"][0],
                                                                          dev_name=datasets[set_name]["filenames"][1],
                                                                          test_name=datasets[set_name]["filenames"][2],
                                                                          has_holdtarg=False)
    _, _, MPQAtest, TEXT1, _, _ = parser.parse_input_files(BATCH_SIZE, EMBEDDING_DIM, using_GPU,
                                                           filepath=datasets[set_name]["filepath"],
                                                           train_name=datasets[set_name]["filenames"][0],
                                                           dev_name=datasets[set_name]["filenames"][1],
                                                           test_name=datasets[set_name]["filenames"][3],
                                                           has_holdtarg=False)

    assert len(TEXT.vocab.stoi) == len(TEXT1.vocab.stoi)
    assert TEXT.vocab.vectors.equal(TEXT1.vocab.vectors)

    word_to_ix = TEXT.vocab.stoi
    ix_to_word = TEXT.vocab.itos

    VOCAB_SIZE = len(word_to_ix)

    word_embeds = TEXT.vocab.vectors

    model = MODEL(NUM_LABELS, VOCAB_SIZE,
                  EMBEDDING_DIM, HIDDEN_DIM, word_embeds,
                  NUM_POLARITIES, BATCH_SIZE, DROPOUT_RATE,
                  max_co_occurs=MAX_CO_OCCURS,
                  ablations=ABLATIONS)

    if MODEL == Model:  # if baseline...
        model = Model(NUM_LABELS, VOCAB_SIZE,
                      EMBEDDING_DIM, HIDDEN_DIM, word_embeds,
                      NUM_POLARITIES, BATCH_SIZE, DROPOUT_RATE)

    print("num params = ")
    print(len(model.state_dict()))

    for epoch in epochs:
        model.load_state_dict(torch.load(save_name(epochs)))

        # Move the model to the GPU if available
        if using_GPU:
            model = model.cuda()

        print("evaluating epoch " + str(epoch) + "...")
        train_score, train_acc = evaluate(model, word_to_ix, ix_to_word, Xtrain, using_GPU)
        print("    train f1 scores = " + str(train_score))
        dev_score, dev_acc = evaluate(model, word_to_ix, ix_to_word, Xdev, using_GPU)
        print("    dev f1 scores = " + str(dev_score))
        ACL_test_score, ACL_test_acc = evaluate(model, word_to_ix, ix_to_word, ACLtest, using_GPU)
        print("    ACL test f1 scores = " + str(ACL_test_score))
        MPQA_test_score, MPQA_test_acc = evaluate(model, word_to_ix, ix_to_word, MPQAtest, using_GPU)
        print("    MPQA test f1 scores = " + str(MPQA_test_score))


if __name__ == "__main__":
    main()
