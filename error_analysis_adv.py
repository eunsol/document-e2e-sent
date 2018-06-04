import torch
import data_processor as parser
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor, EndpointSpanExtractor
from baseline_model_GPU import Model
from advanced_model_1 import Model1
import json

NUM_LABELS = 3
# convention: [NEG, NULL, POS]
epochs = 13
EMBEDDING_DIM = 50
MAX_CO_OCCURS = 10
HIDDEN_DIM = EMBEDDING_DIM
NUM_POLARITIES = 6
DROPOUT_RATE = 0.2
using_GPU = torch.cuda.is_available()
threshold = torch.log(torch.FloatTensor([0.5, 0.1, 0.5]))
if using_GPU:
    threshold = threshold.cuda()

set_name = "C"
datasets = {"A": {"filepath": "./data/new_annot/feature",
                  "filenames": ["new_train.json", "acl_dev_eval_new.json", "acl_test_new.json"],
                  "weights": torch.FloatTensor([0.8, 1.825, 1]),
                  "batch": 10},
            "B": {"filepath": "./data/new_annot/trainsplit_holdtarg",
                  "filenames": ["train.json", "dev.json", "test.json"],
                  "weights": torch.FloatTensor([0.77, 1.766, 1]),
                  "batch": 10},
            "C": {"filepath": "./data/final",
                  "filenames": ["C_train.json", "acl_dev_eval.json", "acl_test.json"],
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
                  "filenames": ["F_train.json", "acl_dev_eval.json", "acl_test.json"],
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
                  "weights": torch.FloatTensor([1, 0.054569, 1.0055]),
                  "batch": 80},
            "no_co_occurs": {"filepath": "./data/no_co_occurs",
                  "filenames": ["F_train.json", "acl_dev_eval_new.json", "acl_test_new.json"],
                  "weights": torch.FloatTensor([1, 0.054569, 1.0055]),
                  "batch": 80}
            }

BATCH_SIZE = 1


def decode(word_indices, ix_to_word):
    words = [ix_to_word[index] for index in word_indices.data[0]]
    return words


def main():
    _, dev_data, _, TEXT, DOCID, _ = parser.parse_input_files(BATCH_SIZE, EMBEDDING_DIM, using_GPU,
                                                              filepath=datasets[set_name]["filepath"],
                                                              train_name=datasets[set_name]["filenames"][0],
                                                              dev_name=datasets[set_name]["filenames"][1],
                                                              test_name=datasets[set_name]["filenames"][2],
                                                              has_holdtarg=True, dev_batch_size=1)

    word_to_ix = TEXT.vocab.stoi
    ix_to_word = TEXT.vocab.itos

    print(len(ix_to_word))
    assert ix_to_word == TEXT.vocab.itos
    print("wow!")

    VOCAB_SIZE = len(word_to_ix)

    word_embeds = TEXT.vocab.vectors
    ix_to_docid = DOCID.vocab.itos

    '''
    model = Model(NUM_LABELS, VOCAB_SIZE,
                  EMBEDDING_DIM, HIDDEN_DIM, word_embeds,
                  NUM_POLARITIES, BATCH_SIZE, DROPOUT_RATE)
    model.load_state_dict(torch.load("./model_states/baseline_" + set_name + "_" + str(epochs) + ".pt"))
    '''
    model = Model1(NUM_LABELS, VOCAB_SIZE,
                   EMBEDDING_DIM, HIDDEN_DIM, word_embeds,
                   NUM_POLARITIES, BATCH_SIZE, DROPOUT_RATE,
                   max_co_occurs=MAX_CO_OCCURS)
    model.load_state_dict(torch.load("./model_states/final/" + set_name + "/span_attentive/adv_" + str(epochs) + ".pt"))
    # '''

    print("num params = ")
    print(len(model.state_dict()))
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
        (words, lengths), polarity, label = batch.text, batch.polarity, batch.label
        holder_targets = batch.holder_target
        (holders, holder_lengths) = batch.holder_index
        (targets, target_lengths) = batch.target_index
        co_occur_feature = batch.co_occurrences
        holder_rank, target_rank = batch.holder_rank, batch.target_rank
        sent_classify = batch.sent_classify

        docid = batch.docid
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()
        model.batch_size = len(label.data)  # set batch size
        # Step 3. Run our forward pass.
        '''
        log_probs, _ = model(words, polarity, holder_targets, lengths)
        '''
        log_probs = model(words, polarity, None, lengths,
                          holders, targets, holder_lengths, target_lengths,
                          co_occur_feature=co_occur_feature,
                          holder_rank=holder_rank, target_rank=target_rank,
                          sent_classify=sent_classify)  # log probs: batch_size x 3
        # '''
        pred_label = log_probs.data.max(1)[1]  # torch.ones(len(log_probs), dtype=torch.long)
        '''
        pred_label = torch.ones(len(log_probs), dtype=torch.long)
        if using_GPU:
            pred_label = pred_label.cuda()
        pred_label[log_probs[:, 2] + 0.02 > log_probs[:, 0]] = 2  # classify more as positive
        pred_label[log_probs[:, 0] > log_probs[:, 2] + 0.02] = 0
        pred_label[log_probs[:, 1] > threshold[1]] = 1  # predict is 1 if even just > 10% certainty
        '''
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

    with open("./error_analysis/" + set_name + "/wrong_docs_dev_adv.json", "w") as wf:
        for line in texts:
            json.dump(line, wf)
            wf.write("\n")
    with open("./error_analysis/" + set_name + "/right_docs_dev_adv.json", "w") as wf:
        for line in right_texts:
            json.dump(line, wf)
            wf.write("\n")


if __name__ == "__main__":
    main()
