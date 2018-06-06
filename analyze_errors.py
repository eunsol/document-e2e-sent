import json
import random
import numpy as np

wrong_filenames = ["./error_analysis/C/wrong_docs_dev.json", "./error_analysis/C/wrong_docs_dev_baseline.json",
                   "./error_analysis/C/wrong_docs_dev_adv.json"]
right_filenames = ["./error_analysis/C/right_docs_dev.json", "./error_analysis/C/right_docs_dev_baseline.json",
                   "./error_analysis/C/right_docs_dev_adv.json"]
docs_file = ["./data/final/acl_dev_eval.json", "./data/final/acl_dev_eval.json", "./data/final/acl_dev_eval.json"]
'''
#wrong_filenames = ["./error_analysis/wrong_docs_has_1s.json", "./error_analysis/wrong_docs_has_1s_mpqa.json",
                   "./error_analysis/added_mention_features/wrong_docs.json",
                   "./error_analysis/added_mention_features/wrong_docs_mpqa.json",
                   "./error_analysis/added_mention_features/wrong_docs_acl_test.json"]
right_filenames = ["./error_analysis/right_docs_has_1s.json", "./error_analysis/right_docs_has_1s_mpqa.json",
                   "./error_analysis/added_mention_features/right_docs.json",
                   "./error_analysis/added_mention_features/right_docs_mpqa.json",
                   "./error_analysis/added_mention_features/right_docs_acl_test.json"]
docs_file = ["./data/new_annot/feature/acl_dev_eval_new.json", "./data/new_annot/feature/mpqa_new.json",
             "./data/new_annot/feature/acl_dev_eval_new.json", "./data/new_annot/feature/mpqa_new.json",
             "./data/new_annot/feature/acl_test_new.json"]
'''
models_map = {1: "baseline", 2: "adv"}

file = 1
model_to_evaluate = "classify" #  models_map[file]

def make_key(holder_inds, target_inds):
    key = ""
    key += str(holder_inds) + "; "
    key += str(target_inds)
    return key


docs_to_labels = {}
doc_probs_no_sent_to_act_label = {}
doc_probs_no_sent_to_pred_label = {}
ex_to_pred = {}
doc_to_pairinds_to_preds = {}

is_0_pred_1 = []
is_0_pred_2 = []
is_1_pred_0 = []
is_1_pred_2 = []
is_2_pred_0 = []
is_2_pred_1 = []
with open(wrong_filenames[file], "r") as rf:
    for l in rf:
        line = json.loads(l)
        if line["docid"] not in docs_to_labels:
            docs_to_labels[line["docid"]] = {"probs": [], "pred": [], "acts": []}
            doc_probs_no_sent_to_act_label[line["docid"]] = {}
            doc_probs_no_sent_to_pred_label[line["docid"]] = {}
        docs_to_labels[line["docid"]]["pred"].append(line["prediction"])
        docs_to_labels[line["docid"]]["acts"].append(line["actual"])
        docs_to_labels[line["docid"]]["probs"].append(line["probabilities"])
        sort_by = line["probabilities"][1]  # max(line["probabilities"][0], line["probabilities"][1])
        doc_probs_no_sent_to_act_label[line["docid"]][sort_by] = line["actual"]
        doc_probs_no_sent_to_pred_label[line["docid"]][sort_by] = line["prediction"]
        if not line["docid"] in doc_to_pairinds_to_preds:
            doc_to_pairinds_to_preds[line["docid"]] = {}
        doc_to_pairinds_to_preds[line["docid"]][make_key(line["holders"], line["targets"])] = line["prediction"]
        ex_to_pred[l] = line["prediction"]


is_0_pred_0 = []
is_1_pred_1 = []
is_2_pred_2 = []
count_0 = 0
with open(right_filenames[file], "r") as rf:
    for l in rf:
        line = json.loads(l)
        if line["actual"] == 0:
            count_0 += 1
        if line["docid"] not in docs_to_labels:
            docs_to_labels[line["docid"]] = {"pred": [], "acts": []}
            doc_probs_no_sent_to_act_label[line["docid"]] = {}
            doc_probs_no_sent_to_pred_label[line["docid"]] = {}
        docs_to_labels[line["docid"]]["pred"].append(line["prediction"])
        docs_to_labels[line["docid"]]["acts"].append(line["actual"])
        sort_by = line["probabilities"][1]  # max(line["probabilities"][0], line["probabilities"][1])
        doc_probs_no_sent_to_act_label[line["docid"]][sort_by] = line["actual"]
        doc_probs_no_sent_to_pred_label[line["docid"]][sort_by] = line["prediction"]
        if not line["docid"] in doc_to_pairinds_to_preds:
            doc_to_pairinds_to_preds[line["docid"]] = {}
        doc_to_pairinds_to_preds[line["docid"]][make_key(line["holders"], line["targets"])] = line["prediction"]
        ex_to_pred[l] = line["prediction"]

with open("./error_analysis/C/acl_dev_eval.json", "r") as rf:
    for l in rf:
        line = json.loads(l)
        if line["label"] == 0:
            if line[model_to_evaluate] == 0:
                is_0_pred_0.append(line)
            if line[model_to_evaluate] == 1:
                is_0_pred_1.append(line)
            if line[model_to_evaluate] == 2:
                is_0_pred_2.append(line)
        if line["label"] == 1:
            if line[model_to_evaluate] == 0:
                is_1_pred_0.append(line)
            if line[model_to_evaluate] == 1:
                is_1_pred_1.append(line)
            if line[model_to_evaluate] == 2:
                is_1_pred_2.append(line)
        if line["label"] == 2:
            if line[model_to_evaluate] == 0:
                is_2_pred_0.append(line)
            if line[model_to_evaluate] == 1:
                is_2_pred_1.append(line)
            if line[model_to_evaluate] == 2:
                is_2_pred_2.append(line)

#  print(docs_to_labels)


#  print(docs_to_labels)

for doc in doc_probs_no_sent_to_act_label:
    print(doc)
    ordered_probs = []
    ordered_labels = []
    ordered_preds = []
    for probs in sorted(doc_probs_no_sent_to_act_label[doc]):
        if doc_probs_no_sent_to_act_label[doc][probs] != 1:
            ordered_labels.append(doc_probs_no_sent_to_act_label[doc][probs])
            ordered_preds.append(doc_probs_no_sent_to_pred_label[doc][probs])
            ordered_probs.append(probs)
    print(ordered_labels)
    print(ordered_preds)


docs_to_label_counts = {}
for doc in docs_to_labels:
    if doc not in docs_to_label_counts:
        docs_to_label_counts[doc] = {"acts": [0, 0, 0], "pred": [0, 0, 0]}
    for act_label in docs_to_labels[doc]["acts"]:
        docs_to_label_counts[doc]["acts"][act_label] += 1
    for pred_label in docs_to_labels[doc]["pred"]:
        docs_to_label_counts[doc]["pred"][pred_label] += 1


def analyze_probabilities():  # for thresholds
    print("analyzing...")
    probability_1s = []
    probability_preds = [[], [], []]
    for entry in is_1_pred_0:
        probability_1s.append(entry["probabilities"][1])
        probability_preds[0].append(entry["probabilities"][0])
    print(sum([probability_1s[i] > 0.02 for i in range(len(probability_1s))]))
    print(sum(probability_1s) / len(probability_1s))
    #  print(sum([probability_preds[0][i] > 0.8 for i in range(len(probability_preds[0]))]))
    print(sum(probability_preds[0]) / len(probability_preds[0]))

    print()
    for entry in is_1_pred_2:
        probability_1s.append(entry["probabilities"][1])
        probability_preds[2].append(entry["probabilities"][2])
    print(sum([probability_1s[i] > 0.02 for i in range(len(probability_1s))]))
    print(sum(probability_1s) / len(probability_1s))
    print(sum(probability_preds[2]) / len(probability_preds[2]))

    print("")
    probabilities = [[], [], []]
    for entry in is_0_pred_0:
        probabilities[0].append(entry["probabilities"][0])
        probabilities[1].append(entry["probabilities"][1])
        probabilities[2].append(entry["probabilities"][2])
    #  print(sum([probabilities[0][i] > 0.8 for i in range(len(probabilities[0]))]))
    print(sum(probabilities[0]) / len(probabilities[0]))
    print(sum([probabilities[1][i] > 0.02 for i in range(len(probabilities[1]))]))
    print(sum(probabilities[1]) / len(probabilities[1]))
    print(sum(probabilities[2]) / len(probabilities[2]))

    print()
    probabilities = [[], [], []]
    for entry in is_2_pred_2:
        probabilities[0].append(entry["probabilities"][0])
        probabilities[1].append(entry["probabilities"][1])
        probabilities[2].append(entry["probabilities"][2])
    print(sum(probabilities[0]) / len(probabilities[0]))
    print(sum(probabilities[1]) / len(probabilities[1]))
    print(sum(probabilities[2]) / len(probabilities[2]))

    print()
    probabilities = [[], [], []]
    for entry in is_1_pred_1:
        probabilities[0].append(entry["probabilities"][0])
        probabilities[1].append(entry["probabilities"][1])
        probabilities[2].append(entry["probabilities"][2])
    print(sum(probabilities[0]) / len(probabilities[0]))
    print(sum(probabilities[1]) / len(probabilities[1]))
    print(sum(probabilities[2]) / len(probabilities[2]))
    print("done probabilities")
    print()


#  analyze_probabilities()  # for thresholds


for doc in docs_to_labels:
    acts = [docs_to_labels[doc]["acts"].count(i) for i in range(3)]
    pred = [docs_to_labels[doc]["pred"].count(i) for i in range(3)]

total_is_0 = len(is_0_pred_0) + len(is_0_pred_1) + len(is_0_pred_2)
total_is_1 = len(is_1_pred_0) + len(is_1_pred_1) + len(is_1_pred_2)
total_is_2 = len(is_2_pred_0) + len(is_2_pred_1) + len(is_2_pred_2)
print("# examples of each label (total): ")
print(total_is_0)
print(total_is_1)
print(total_is_2)
print()

print("# of examples labelled 0 but predicted...: ")
print(len(is_0_pred_0))  # / total_is_0)
print(len(is_0_pred_1))  # / total_is_0)
print(len(is_0_pred_2))  # / total_is_0)
print()
print("# of examples labelled 1 but predicted...: ")
print(len(is_1_pred_0))  # / total_is_1)
print(len(is_1_pred_1))  # / total_is_1)
print(len(is_1_pred_2))  # / total_is_1)
print()
print("# of examples labelled 2 but predicted...: ")
print(len(is_2_pred_0))  # / total_is_2)
print(len(is_2_pred_1))  # / total_is_2)
print(len(is_2_pred_2))  # / total_is_2)
print()

doc_to_text = {}
ex_to_label = {}
doc_to_ht_to_cooccur = {}
doc_to_entinds_to_mention_rank = {}
doc_to_entinds_to_entnames = {}
with open(docs_file[file], "r") as rf:
    for l in rf:
        line = json.loads(l)
        if line["docid"] not in doc_to_text:
            doc_to_text[line["docid"]] = []
            doc_to_text[line["docid"]] = line["token"]
        ex_to_label[l] = line["label"]
        if not line["docid"] in doc_to_ht_to_cooccur:
            doc_to_ht_to_cooccur[line["docid"]] = {}
        if not line["docid"] in doc_to_entinds_to_mention_rank:
            doc_to_entinds_to_mention_rank[line["docid"]] = {}
        if make_key(line["holder_index"], line["target_index"]) in doc_to_ht_to_cooccur[line["docid"]]:
            assert doc_to_ht_to_cooccur[line["docid"]][make_key(line["holder_index"], line["target_index"])] == line[
                "co_occurrences"]
        if not line["docid"] in doc_to_entinds_to_entnames:
            doc_to_entinds_to_entnames[line["docid"]] = {}
        doc_to_ht_to_cooccur[line["docid"]][make_key(line["holder_index"], line["target_index"])] = line[
            "co_occurrences"]
        doc_to_entinds_to_mention_rank[line["docid"]][str(line["holder_index"])] = line["holder_rank"]
        doc_to_entinds_to_mention_rank[line["docid"]][str(line["target_index"])] = line["target_rank"]
        doc_to_entinds_to_entnames[line["docid"]][str(line["holder_index"])] = line["holder"]
        doc_to_entinds_to_entnames[line["docid"]][str(line["target_index"])] = line["target"]


def convert_to_str(list):
    str = ""
    i = 0
    for token in list:
        if token is not "'" and token is not "." and token is not "\"\"" and token is not "," and token is not "''":
            str += " " + token
        else:
            str += token
        i += 1
    return str


def analyze_num_mentions(list1):
    num_mentions_label = [0, 0, 0]
    min_num_mentions_label = [0, 0, 0]
    mentions_label = [[], [], []]
    docs_label = [0, 0, 0]
    num_mentions_label_h = [0, 0, 0]
    mentions_label_h = [[], [], []]
    num_mentions_label_t = [0, 0, 0]
    mentions_label_t = [[], [], []]

    h = "holder_index"
    t = "target_index"
    if list1 == ex_to_pred:
        h = "holders"
        t = "targets"
    for l in list1:
        line = json.loads(l)
        docs_label[list1[l]] += 1

        mentions_label[list1[l]].append(len(line[h]))
        num_mentions_label[list1[l]] += len(line[h])
        mentions_label[list1[l]].append(len(line[t]))
        num_mentions_label[list1[l]] += len(line[t])
        min_num_mentions_label[list1[l]] += min(len(line[h]), len(line[t]))

        mentions_label_h[list1[l]].append(len(line[h]))
        num_mentions_label_h[list1[l]] += len(line[h])
        mentions_label_t[list1[l]].append(len(line[t]))
        num_mentions_label_t[list1[l]] += len(line[t])

    print(docs_label)
    print(num_mentions_label_h)
    print([num_mentions_label_h[i] / docs_label[i] for i in range(3)])
    print(num_mentions_label_t)
    print([num_mentions_label_t[i] / docs_label[i] for i in range(3)])
    print([num_mentions_label[i] / docs_label[i] for i in range(3)])
    print([min_num_mentions_label[i] / docs_label[i] for i in range(3)])
    print()
    for j in range(1, 10):
        print("# of holder mentions = " + str(j))
        print([mentions_label_h[i].count(j) / num_mentions_label_h[i] for i in range(3)])
    print()
    for j in range(1, 10):
        print("# of target mentions = " + str(j))
        print([mentions_label_t[i].count(j) / num_mentions_label_t[i] for i in range(3)])
    print()
    print("# of holder or target mentions = 1")
    to_print = [0, 0, 0]
    for i in range(3):
        holder_or_target_is_1 = \
            [mentions_label_h[i][j] == 1 or mentions_label_t[i][j] == 1 for j in range(len(mentions_label_h[i]))]
        to_print[i] = holder_or_target_is_1.count(True) / num_mentions_label[i]
    print(to_print)
    print()
    for j in range(1, 10):
        print("# of total mentions = " + str(j))
        print([mentions_label[i].count(j) / num_mentions_label[i] for i in range(3)])


def ave_same_sentence_features(exs):
    co_occurs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for ex in exs:
        ind = int(doc_to_ht_to_cooccur[ex["docid"]][make_key(ex["holders"], ex["targets"])])
        if ind >= 10:
            ind = 9
        co_occurs[ind] += 1
    ave_co_occurs = 0
    for i in range(0, len(co_occurs)):
        ave_co_occurs += (i + 1) * co_occurs[i]
    ave_co_occurs /= sum(co_occurs)
    return ave_co_occurs


def ave_rank_features(exs):
    holder_rank_distr = [0, 0, 0, 0, 0]
    target_rank_distr = [0, 0, 0, 0, 0]
    for ex in exs:
        holder_rank = int(doc_to_entinds_to_mention_rank[ex["docid"]][str(ex["holders"])])
        target_rank = int(doc_to_entinds_to_mention_rank[ex["docid"]][str(ex["targets"])])
        if holder_rank >= 6:
            holder_rank = 5
        if target_rank >= 6:
            target_rank = 5
        holder_rank -= 1
        target_rank -= 1
        holder_rank_distr[holder_rank] += 1
        target_rank_distr[target_rank] += 1
    print([holder_rank_distr[i] / sum(holder_rank_distr) for i in range(len(holder_rank_distr))])
    print([target_rank_distr[i] / sum(target_rank_distr) for i in range(len(target_rank_distr))])
    ave_holder_rank = sum(holder_rank_distr) / len(holder_rank_distr)
    ave_target_rank = sum(target_rank_distr) / len(target_rank_distr)
    return ave_holder_rank, ave_target_rank


def save_examples(exs, filename, condense_factor):
    all_exs = []
    for example in exs:
        docid = example["docid"]
        token = doc_to_text[docid].copy()
        for holder_ind in example["holders"]:
            token[holder_ind[0]] = "\\textbf{" + token[holder_ind[0]]
            token[holder_ind[1]] = token[holder_ind[1]] + "}"
        for target_ind in example["targets"]:
            token[target_ind[0]] = "\\underline{" + token[target_ind[0]]
            token[target_ind[1]] = token[target_ind[1]] + "}"
        all_exs.append(convert_to_str(token))

    random.shuffle(all_exs)
    all_exs = all_exs[0:int(condense_factor * len(all_exs))]
    print(len(all_exs))
    with open(filename, "w") as wf:
        for line in all_exs:
            wf.write("\\par" + line)
            wf.write("\n")
    '''
    holder_inds = example["holders"][0]
    target_inds = example["targets"][0]
    holder = convert_to_str(token[holder_inds[0]:holder_inds[1] + 1])
    target = convert_to_str(token[target_inds[0]:target_inds[1] + 1])
    holder2 = ""
    target2 = ""
    if len(example["holders"]) > 1:
        holder_inds2 = example["holders"][1]
        holder2 = convert_to_str(token[holder_inds2[0]:holder_inds2[1] + 1])
    if len(example["targets"]) > 1:
        target_inds2 = example["targets"][1]
        target2 = convert_to_str(token[target_inds2[0]:target_inds2[1] + 1])
    print(str(docid) + " " + str(holder) + "," + str(holder2) + "      " + str(target) + "," + str(target2))
    '''


pairs_to_labels_train = {}
with open("./data/new_annot/feature/F_train.json", "r") as rf:
    for l in rf:
        line = json.loads(l)
        ht_key = str(line["holder"]) + "; " + str(line["target"])
        ht_key_reversed = str(line["target"]) + "; " + str(line["holder"])
        if ht_key not in pairs_to_labels_train:
            pairs_to_labels_train[ht_key] = []
            pairs_to_labels_train[ht_key_reversed] = []
        pairs_to_labels_train[ht_key].append(line["label"])
        pairs_to_labels_train[ht_key_reversed].append(line["label"])


def pairs_classified():
    pairs_to_labels = {}

    for ex in ex_to_label:
        ex_map = json.loads(ex)
        ht_inds_key = make_key(ex_map["holder_index"], ex_map["target_index"])
        pred = doc_to_pairinds_to_preds[ex_map["docid"]][ht_inds_key]
        pair_key = ex_map["holder"] + "; " + ex_map["target"]
        pair_key_reversed = ex_map["target"] + "; " + ex_map["holder"]
        pair_key = pair_key.lower()
        pair_key_reversed = pair_key_reversed.lower()
        if pair_key not in pairs_to_labels:
            pairs_to_labels[pair_key] = {"actual": [], "pred": []}
            pairs_to_labels[pair_key_reversed] = {"actual": [], "pred": []}
        pairs_to_labels[pair_key]["actual"].append(ex_to_label[ex])
        pairs_to_labels[pair_key_reversed]["actual"].append(ex_to_label[ex])
        pairs_to_labels[pair_key]["pred"].append(pred)
        pairs_to_labels[pair_key_reversed]["pred"].append(pred)

    pairs_to_actual_condensed = {}
    pairs_to_actual_uniform = {}
    pairs_to_actual_non_uniform = {}

    pairs_to_preds_condensed = {}
    pairs_to_preds_uniform = {}
    pairs_to_preds_non_uniform = {}
    for pairs in pairs_to_labels:
        if len(pairs_to_labels[pairs]["actual"]) > 2:
            if pairs in pairs_to_labels_train:
                print(str(pairs) + " " + str(pairs_to_labels[pairs]) + " " + str(pairs_to_labels_train[pairs]))
            pairs_to_actual_condensed[pairs] = pairs_to_labels[pairs]["actual"]
            pairs_to_preds_condensed[pairs] = pairs_to_labels[pairs]["pred"]
            actual_is_uniform = True
            pred_is_uniform = True
            for i in range(len(pairs_to_labels[pairs]["actual"])):
                if pairs_to_labels[pairs]["actual"][i] != pairs_to_labels[pairs]["actual"][0]:
                    actual_is_uniform = False
                if pairs_to_labels[pairs]["pred"][i] != pairs_to_labels[pairs]["pred"][0]:
                    pred_is_uniform = False
            if actual_is_uniform:
                pairs_to_actual_uniform[pairs] = pairs_to_labels[pairs]["actual"]
            else:
                pairs_to_actual_non_uniform[pairs] = pairs_to_labels[pairs]["actual"]
            if pred_is_uniform:
                pairs_to_preds_uniform[pairs] = pairs_to_labels[pairs]["pred"]
            else:
                pairs_to_preds_non_uniform[pairs] = pairs_to_labels[pairs]["pred"]
    print(len(pairs_to_labels) / 2)
    print(len(pairs_to_actual_condensed) / 2)
    print("actual uniform pairs = " + str(len(pairs_to_actual_uniform) / 2))
    pair_breakdown = [0, 0, 0]
    for pair in pairs_to_actual_uniform:
        pair_breakdown[pairs_to_actual_uniform[pair][0]] += 0.5
    print("      " + str(pair_breakdown))
    print("actual non-uniform pairs = " + str(len(pairs_to_actual_non_uniform) / 2))
    #  print("predicted pairs = " + str(len(pairs_to_preds)))
    print("predicted uniform pairs = " + str(len(pairs_to_preds_uniform) / 2))
    pair_breakdown_preds = [0, 0, 0]
    for pair in pairs_to_preds_uniform:
        pair_breakdown_preds[pairs_to_preds_uniform[pair][0]] += 0.5
    print("      " + str(pair_breakdown_preds))
    print("predicted non-uniform pairs = " + str(len(pairs_to_preds_non_uniform) / 2))

    num_uniform_same = [0, 0, 0]
    num_uniform_diff = [0, 0, 0]
    num_uniform_diff_pred = [0, 0, 0]
    num_non_uniform = [0, 0, 0]
    for pair in pairs_to_actual_uniform:
        if pair in pairs_to_preds_uniform:
            if pairs_to_preds_uniform[pair][0] == pairs_to_actual_uniform[pair][0]:
                num_uniform_same[pairs_to_actual_uniform[pair][0]] += 0.5
            else:
                num_uniform_diff[pairs_to_actual_uniform[pair][0]] += 0.5
                num_uniform_diff_pred[pairs_to_preds_uniform[pair][0]] += 0.5
        else:
            num_non_uniform[pairs_to_actual_uniform[pair][0]] += 0.5
    print("# of uniform pairs predicted the right way = " + str(num_uniform_same))  # [num_uniform_same[i] / sum(num_uniform_same) for i in range(len(num_uniform_same))]))
    print("# of uniform pairs predicted the wrong way = " + str(num_uniform_diff))  # [num_uniform_diff[i] / sum(num_uniform_diff) for i in range(len(num_uniform_diff))]))
    print("# of uniform pairs predicted the wrong way = " + str(num_uniform_diff_pred))  # [num_uniform_diff_pred[i] / sum(num_uniform_diff_pred) for i in range(len(num_uniform_diff))]))
    print("# of uniform pairs predicted non-uniform = " + str(num_non_uniform))  # [num_non_uniform[i] / sum(num_non_uniform) for i in range(len(num_non_uniform))]))


def count_num_mentions(filename):
    holder_lens = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    target_lens = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    min_lens = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    with open(filename, "r", encoding="latin1") as rf:
        for line in rf:
            annot = json.loads(line)
            min_len = min(len(annot["holder_index"]) - 1, len(annot["target_index"]) - 1)
            if len(annot["holder_index"]) <= len(holder_lens[annot["label"]]):
                holder_lens[annot["label"]][len(annot["holder_index"]) - 1] += 1
            else:
                holder_lens[annot["label"]][len(holder_lens) - 1] += 1
                #  print(len(annot["holder_index"]) - 1)
            if len(annot["target_index"]) <= len(target_lens[annot["label"]]):
                target_lens[annot["label"]][len(annot["target_index"]) - 1] += 1
            else:
                target_lens[annot["label"]][len(target_lens) - 1] += 1
                #  print(len(annot["target_index"]) - 1)
            if min_len < len(min_lens[annot["label"]]):
                min_lens[annot["label"]][min_len] += 1
            else:
                min_lens[annot["label"]][len(min_lens) - 1] += 1
    '''
    print([holder_lens[0][i] / (holder_lens[0][i] + holder_lens[1][i] + holder_lens[2][i]) if (holder_lens[0][i] +
                                                                                               holder_lens[1][i] +
                                                                                               holder_lens[2][
                                                                                                   i]) > 0 else 0 for i
           in range(len(holder_lens[0]))])
    print([holder_lens[1][i] / (holder_lens[0][i] + holder_lens[1][i] + holder_lens[2][i]) if (holder_lens[0][i] +
                                                                                               holder_lens[1][i] +
                                                                                               holder_lens[2][
                                                                                                   i]) > 0 else 0 for i
           in range(len(holder_lens[0]))])
    print([holder_lens[2][i] / (holder_lens[0][i] + holder_lens[1][i] + holder_lens[2][i]) if (holder_lens[0][i] +
                                                                                               holder_lens[1][i] +
                                                                                               holder_lens[2][
                                                                                                   i]) > 0 else 0 for i
           in range(len(holder_lens[0]))])
    print([target_lens[0][i] / (target_lens[0][i] + target_lens[1][i] + target_lens[2][i]) if (target_lens[0][i] +
                                                                                               target_lens[1][i] +
                                                                                               target_lens[2][
                                                                                                   i]) > 0 else 0 for i
           in range(len(target_lens[0]))])
    print([target_lens[1][i] / (target_lens[0][i] + target_lens[1][i] + target_lens[2][i]) if (target_lens[0][i] +
                                                                                               target_lens[1][i] +
                                                                                               target_lens[2][
                                                                                                   i]) > 0 else 0 for i
           in range(len(target_lens[0]))])
    print([target_lens[2][i] / (target_lens[0][i] + target_lens[1][i] + target_lens[2][i]) if (target_lens[0][i] +
                                                                                               target_lens[1][i] +
                                                                                               target_lens[2][
                                                                                                   i]) > 0 else 0 for i
           in range(len(target_lens[0]))])
    '''
    print([min_lens[0][i] / (min_lens[0][i] + min_lens[1][i] + min_lens[2][i]) if (min_lens[0][i] +
                                                                                   min_lens[1][i] +
                                                                                   min_lens[2][
                                                                                       i]) > 0 else 0 for i
           in range(len(min_lens[0]))])
    print([min_lens[1][i] / (min_lens[0][i] + min_lens[1][i] + min_lens[2][i]) if (min_lens[0][i] +
                                                                                   min_lens[1][i] +
                                                                                   min_lens[2][
                                                                                       i]) > 0 else 0 for i
           in range(len(min_lens[0]))])
    print([min_lens[2][i] / (min_lens[0][i] + min_lens[1][i] + min_lens[2][i]) if (min_lens[0][i] +
                                                                                   min_lens[1][i] +
                                                                                   min_lens[2][
                                                                                       i]) > 0 else 0 for i
           in range(len(min_lens[0]))])
    # print([holder_lens[i] / sum(holder_lens) for i in range(len(holder_lens))])
    # print([target_lens[i] / sum(target_lens) for i in range(len(target_lens))])


def write_to_file():
    all_data = []
    with open("./error_analysis/C/acl_dev_eval.json", "r", encoding="latin1") as rf:
        for line in rf:
            annot = json.loads(line)
            pred = doc_to_pairinds_to_preds[annot["docid"]][make_key(annot["holder_index"], annot["target_index"])]
            annot[model_to_evaluate] = pred
            all_data.append(annot)

    with open("./error_analysis/C/acl_dev_eval.json", "w", encoding="latin1") as wf:
        for annot in all_data:
            json.dump(annot, wf)
            wf.write("\n")


def count_label_freqs_per_co_occur():
    co_occur_to_label_distr = {}
    co_occur_to_preds_distr = {}
    max_co_occur = 0
    square = [[0, 0, 0],
              [0, 0, 0],
              [0, 0, 0]]
    with open("./error_analysis/C/acl_dev_eval.json", "r", encoding="latin1") as rf:
        for line in rf:
            annot = json.loads(line)
            # Note same set of co-occurrences in each
            if annot["co_occurrences"] not in co_occur_to_label_distr:
                co_occur_to_label_distr[annot["co_occurrences"]] = [0, 0, 0]
                co_occur_to_preds_distr[annot["co_occurrences"]] = [0, 0, 0]
                if annot["co_occurrences"] > max_co_occur:
                    max_co_occur = annot["co_occurrences"]
            co_occur_to_label_distr[annot["co_occurrences"]][annot["label"]] += 1
            co_occur_to_preds_distr[annot["co_occurrences"]][annot[model_to_evaluate]] += 1

            if annot["co_occurrences"] == 1:
                square[annot["classify"]][annot["label"]] += 1

    print(square)

    max_co_occur += 1
    co_occur_to_labels = [i for i in range(max_co_occur)]
    co_occur_to_preds = [i for i in range(max_co_occur)]

    for i in range(max_co_occur):
        if i in co_occur_to_label_distr:
            co_occur_to_labels[i] = co_occur_to_label_distr[i]
            co_occur_to_preds[i] = co_occur_to_preds_distr[i]
        else:
            co_occur_to_labels[i] = [0, 0, 0]
            co_occur_to_preds[i] = [0, 0, 0]

    print(co_occur_to_label_distr)
    for co_occur in co_occur_to_labels:
        print(str(co_occur[0]) + "\t" + str(co_occur[1]) + "\t" + str(co_occur[2]))
    print(co_occur_to_labels)
    print(co_occur_to_preds_distr)
    for co_occur in co_occur_to_preds:
        print(str(co_occur[0]) + "\t" + str(co_occur[1]) + "\t" + str(co_occur[2]))
    print(co_occur_to_preds)


def main():
    #'APW_ENG_20100521.0054': {'acts': [0, 79, 11]
    # 'XIN_ENG_20100915.0131': {'acts': [0, 37, 5], 'pred': [1, 13, 28]}
    print(docs_to_label_counts)
    '''
    print(ave_same_sentence_features(is_0_pred_0))
    print(ave_same_sentence_features(is_0_pred_1))
    print(ave_same_sentence_features(is_0_pred_2))

    print(ave_same_sentence_features(is_1_pred_0))
    print(ave_same_sentence_features(is_1_pred_1))
    print(ave_same_sentence_features(is_1_pred_2))

    print(ave_same_sentence_features(is_2_pred_0))
    print(ave_same_sentence_features(is_2_pred_1))
    print(ave_same_sentence_features(is_2_pred_2))
    '''
    '''
    print(ave_rank_features(is_0_pred_0))
    print(ave_rank_features(is_0_pred_1))
    print(ave_rank_features(is_0_pred_2))

    print(ave_rank_features(is_1_pred_0))
    print(ave_rank_features(is_1_pred_1))
    print(ave_rank_features(is_1_pred_2))

    print(ave_rank_features(is_2_pred_0))
    print(ave_rank_features(is_2_pred_1))
    print(ave_rank_features(is_2_pred_2))
    '''
    '''
    print("num mentions")
    analyze_num_mentions(ex_to_label)
    print()
    analyze_num_mentions(ex_to_pred)
    print()
    '''
    #  save_examples(is_0_pred_1, "./error_analysis/acl_guessed_0_is_1_condense.txt", 1)
    #  save_examples(is_1_pred_2, "./error_analysis/C/guessed_2_is_1_condense.txt", 0.15)
    # save_examples(is_1_pred_0, "./error_analysis/C/guessed_0_is_1_condense.txt", 0.3)
    # save_examples(is_2_pred_0, "./error_analysis/C/guessed_2_is_0_condense.txt", 1)

    #  save_examples(is_2_pred_1, "./error_analysis/acl_guessed_2_is_1_condense.txt", 0.5)
    #  pairs_classified()
    print()
    #  write_to_file()
    count_label_freqs_per_co_occur()
    '''
    count_num_mentions("./data/new_annot/feature/mpqa_new.json")
    #  count_num_mentions("./data/new_annot/feature/new_train.json")
    count_num_mentions("./data/new_annot/feature/F_train.json")
    count_num_mentions("./data/new_annot/feature/acl_dev_eval_new.json")
    '''


if __name__ == "__main__":
    main()

