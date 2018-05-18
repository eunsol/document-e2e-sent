import torch
import data_processor as parser
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor
from advanced_model_1 import Model1
import json

NUM_LABELS = 3
# convention: [NEG, NULL, POS]
epochs = 10
EMBEDDING_DIM = 50
MAX_CO_OCCURS = 10
HIDDEN_DIM = EMBEDDING_DIM
NUM_POLARITIES = 6
DROPOUT_RATE = 0.2
using_GPU = torch.cuda.is_available()
threshold = torch.log(torch.FloatTensor([0.5, 0.1, 0.5]))
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
                  "filenames": ["F_train.json", "acl_dev_eval_new.json", "mpqa_new.json"],
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


def decode(word_indices, ix_to_word):
    words = [ix_to_word[index] for index in word_indices.data[0]]
    return words


def main():
    dev_data, _, _, TEXT, DOCID, _ = parser.parse_input_files(BATCH_SIZE, EMBEDDING_DIM, using_GPU,
                                                              filepath=datasets[set_name]["filepath"],
                                                              train_name=datasets[set_name]["filenames"][1],
                                                              dev_name=datasets[set_name]["filenames"][0],
                                                              test_name=datasets[set_name]["filenames"][2],
                                                              has_holdtarg=True)

    word_to_ix = TEXT.vocab.stoi
    ix_to_word = TEXT.vocab.itos

    VOCAB_SIZE = len(word_to_ix)

    word_embeds = TEXT.vocab.vectors

    model = Model1(NUM_LABELS, VOCAB_SIZE,
                   EMBEDDING_DIM, HIDDEN_DIM, word_embeds,
                   NUM_POLARITIES, BATCH_SIZE, DROPOUT_RATE,
                   max_co_occurs=MAX_CO_OCCURS)

    print("num params = ")
    print(len(model.state_dict()))
    model.load_state_dict(torch.load("./model_states/adv_" + set_name + "_" + str(epochs) + "_mpqa.pt"))
    model.eval()

    # Move the model to the GPU if available
    if using_GPU:
        model = model.cuda()

    print("num batches = " + str(len(dev_data)))
    counter = 0
    preds = []
    probs = []
    acts = []
    texts = []
    right_texts = []
    for batch in dev_data:
        counter += 1
        (words, lengths), polarity, holder_target, label = batch.text, batch.polarity, batch.holder_target, batch.label
        (holders, holder_lengths) = batch.holder_index
        (targets, target_lengths) = batch.target_index
        co_occur_feature = batch.co_occurrences
        docid = batch.docid
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()
        model.batch_size = len(label.data)  # set batch size
        # Step 3. Run our forward pass.
        log_probs = model(words, polarity, holder_target, lengths,
                          holders, targets, holder_lengths, target_lengths,
                          co_occur_feature=co_occur_feature)  # log probs: batch_size x 3
#        pred_label = log_probs.data.max(1)[1]  # torch.ones(len(log_probs), dtype=torch.long)

        pred_label = torch.ones(len(log_probs), dtype=torch.long)
        if using_GPU:
            pred_label = pred_label.cuda()
        pred_label[log_probs[:, 2] + 0.02 > log_probs[:, 0]] = 2  # classify more as positive
        pred_label[log_probs[:, 0] > log_probs[:, 2] + 0.02] = 0
        pred_label[log_probs[:, 1] > threshold[1]] = 1  # predict is 1 if even just > 10% certainty

        if int(pred_label) != -1:
            prob = torch.exp(log_probs)
            probs.append(prob[0].data.cpu().numpy().tolist())
            preds.append(int(pred_label))
            acts.append(int(label))
            entry = {"docid": DOCID.vocab.itos[int(docid)],
                     "holders": holders[0].data.cpu().numpy().tolist(),
                     "targets": targets[0].data.cpu().numpy().tolist(),
                     "probabilities": probs[len(probs) - 1],
                     "prediction": preds[len(preds) - 1],
                     "actual": acts[len(acts) - 1]}
            if int(pred_label) != int(label):
                texts.append(entry)
            else:
                right_texts.append(entry)
        if counter % 100 == 0:
            print(counter)
            print(len(texts))

    print(probs)
    print(preds)
    print(acts)
    with open("./error_analysis/added_mention_features/wrong_docs_thresholds.json", "w") as wf:
        for line in texts:
            json.dump(line, wf)
            wf.write("\n")
    with open("./error_analysis/added_mention_features/right_docs_mpqa_thresholds.json", "w") as wf:
        for line in right_texts:
            json.dump(line, wf)
            wf.write("\n")
    print(texts)


if __name__ == "__main__":
    main()
