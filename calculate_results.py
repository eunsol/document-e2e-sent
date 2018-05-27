import json
import numpy as np

# [precision of has_co_occur, precision of no_co_occur]
train_precision = np.array([[0.31344845630559914, 0.9787164374590701, 0.4006734006734007],
                            [0.4716636197440585, 0.9976072151665747, 0.3248407643312102]])
train_recall = np.array([[0.9492868462757528, 0.5808394869801787, 0.8841010401188707],
                         [0.9214285714285714, 0.9402376615491369, 0.8793103448275862]])
train_f1_scores = np.array([[0.47128245476003144, 0.7290243902439025, 0.551436515291937],
                            [0.6239419588875453, 0.9680732306318374, 0.4744186046511627]])
dev_precision = np.array([[0.2264957264957265, 0.8477088948787062, 0.16666666666666666],
                          [0.02040816326530612, 0.9491289198606272, 0.06382978723404255]])
dev_recall = np.array([[0.5638297872340425, 0.5945179584120983, 0.3372093023255814],
                       [0.041666666666666664, 0.9147078576225655, 0.09230769230769231]])
dev_f1_scores = np.array([[0.3231707317073171, 0.6988888888888889, 0.22307692307692306],
                          [0.027397260273972598, 0.9316005471956226, 0.07547169811320754]])
test_precision = np.array([[0.13114754098360656, 0.8988326848249028, 0.158],
                           [0.03862660944206009, 0.951393005334914, 0.0711743772241993]])
test_recall = np.array([[0.42748091603053434, 0.6877924287537218, 0.3237704918032787],
                        [0.13432835820895522, 0.870862723819859, 0.14814814814814814]])
test_f1_scores = np.array([[0.2007168458781362, 0.779277108433735, 0.2123655913978495],
                           [0.060000000000000005, 0.9093484419263456, 0.09615384615384617]])

filepaths = ["./data/has_co_occurs/", "./data/no_co_occurs/"]
train_file = "F_train.json"
dev_file = "acl_dev_eval_new.json"
test_file = "acl_test_new.json"

train_acts = np.zeros((2, 3))
dev_acts = np.zeros((2, 3))
test_acts = np.zeros((2, 3))
train_act_and_pred = [[], []]
dev_act_and_pred = [[], []]
test_act_and_pred = [[], []]
train_preds = [[], []]
dev_preds = [[], []]
test_preds = [[], []]
'''
for f in range(len(filepaths)):
    filepath = filepaths[f]
    print(filepath)

    with open(filepath + train_file, "r", encoding="latin1") as trf:
        for line in trf:
            annot = json.loads(line)
            train_acts[f][annot["label"]] += 1

    with open(filepath + dev_file, "r", encoding="latin1") as trf:
        for line in trf:
            annot = json.loads(line)
            dev_acts[f][annot["label"]] += 1

    with open(filepath + test_file, "r", encoding="latin1") as trf:
        for line in trf:
            annot = json.loads(line)
            test_acts[f][annot["label"]] += 1

train_act_and_pred = train_recall * train_acts
dev_act_and_pred = dev_recall * dev_acts
test_act_and_pred = test_recall * test_acts

train_preds = train_act_and_pred / train_precision
dev_preds = dev_act_and_pred / dev_precision
test_preds = test_act_and_pred / test_precision

print("Actual label distribution:")
print("    " + str(train_acts))
print("    " + str(dev_acts))
print("    " + str(test_acts))

print("Correctly predicted label distribution:")
print("    " + str(train_act_and_pred))
print("    " + str(dev_act_and_pred))
print("    " + str(test_act_and_pred))

print("Predicted label distribution:")
print("    " + str(train_preds))
print("    " + str(dev_preds))
print("    " + str(test_preds))

total_acts = [[], [], []]
total_acts[0] = train_acts.sum(axis=0)
total_acts[1] = dev_acts.sum(axis=0)
total_acts[2] = test_acts.sum(axis=0)

print(total_acts)

total_act_and_pred = [[], [], []]
total_act_and_pred[0] = train_act_and_pred.sum(axis=0)
total_act_and_pred[1] = dev_act_and_pred.sum(axis=0)
total_act_and_pred[2] = test_act_and_pred.sum(axis=0)

print(total_act_and_pred)

total_preds = [[], [], []]
total_preds[0] = train_preds.sum(axis=0)
total_preds[1] = dev_preds.sum(axis=0)
total_preds[2] = test_preds.sum(axis=0)

print(total_preds)

for i in range(len(total_acts)):
    if i == 0:
        print("Training")
    elif i == 1:
        print("Dev")
    else:
        print("Test")
    acts = total_acts[i]
    act_and_pred = total_act_and_pred[i]
    preds = total_preds[i]

    # Precision = act & pred / pred
    total_precision = act_and_pred / preds
    # Recall = act & pred / act
    total_recall = act_and_pred / acts
    # F1 = 2 * P * R / (P + R)
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall)

    print(total_precision)
    print(total_recall)
    print(total_f1)
'''
filenames = ["acl_dev_eval_new.json", "acl_test_new.json", "mpqa_new.json"]

for fn in filenames:
    print(fn)

    # Compare with results from just sentence
    all_correct_data = [[], [], []]
    all_wrong_data_pred = [[], [], []]
    all_wrong_data_acts = [[], [], []]

    num_correct_data = [0, 0, 0]
    num_wrong_data_pred = [0, 0, 0]
    num_wrong_data_acts = [0, 0, 0]
    for filepath in filepaths:
        with open(filepath + fn, "r", encoding="latin1") as rf:
            for line in rf:
                annot = json.loads(line)
                label = annot["label"]
                pred = annot["classify"]
                if label == pred:
                    all_correct_data[label].append(annot)
                    num_correct_data[label] += 1
                all_wrong_data_pred[pred].append(annot)
                all_wrong_data_acts[label].append(annot)
                num_wrong_data_pred[pred] += 1
                num_wrong_data_acts[label] += 1

    print(num_correct_data)
    print(num_wrong_data_pred)
    print(num_wrong_data_acts)

    precision = [0, 0, 0]
    recall = [0, 0, 0]
    total_f1 = [0, 0, 0]
    for i in range(3):
        print(i)
        if len(all_wrong_data_pred[i]) == 0:
            precision[i] = 0
        else:
            precision[i] = len(all_correct_data[i]) / len(all_wrong_data_pred[i])
        recall[i] = len(all_correct_data[i]) / len(all_wrong_data_acts[i])
        if precision[i] + recall[i] == 0:
            total_f1[i] = 0
        else:
            total_f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
    print(precision)
    print(recall)
    print(total_f1)
