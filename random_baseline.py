import data_processor as parser
import torch
import random
import numpy as np
from sklearn.metrics import f1_score


BATCH_SIZE = 10
EMBEDDING_DIM = 50

def main():
    train_data, dev_data, test_data, TEXT = parser.parse_input_files(BATCH_SIZE, EMBEDDING_DIM, using_GPU=False)

    print()

    pos, neg, null = train(train_data)
    raw_nums = [neg, null, pos]
    print(raw_nums)
    pos = pos / sum(raw_nums)
    neg = neg / sum(raw_nums)
    null = null / sum(raw_nums)
    ratios = [neg, null, pos]
    print(ratios)
    train_accs, train_f1, train_p, train_r = evaluate(train_data, ratios, is_train=True)
    dev_accs, dev_f1, dev_p, dev_r = evaluate(dev_data, ratios)
    test_accs, test_f1, test_p, test_r = evaluate(test_data, ratios)

    print()
    print("Train results")
    print("    f1s = " + str(train_f1) + " " + str(sum(train_f1) / len(train_f1)))
    print("    pre = " + str(train_p))
    print("    rec = " + str(train_r))
    print("    acc = " + str(train_accs))
    print("Dev results")
    print("    f1s = " + str(dev_f1) + " " + str(sum(dev_f1) / len(dev_f1)))
    print("    pre = " + str(dev_p))
    print("    rec = " + str(dev_r))
    print("    acc = " + str(dev_accs))
    print("Test results")
    print("    f1s = " + str(test_f1) + " " + str(sum(test_f1) / len(test_f1)))
    print("    pre = " + str(test_p))
    print("    rec = " + str(test_r))
    print("    acc = " + str(test_accs))

    train_ = [0.8059490084985835, 0.47965738758029974, 0.781630740393627]
    dev_ = [0.5325443786982249, 0.21212121212121213, 0.5454545454545455]
    test_ = [0.6818181818181819, 0.21428571428571427, 0.5692307692307693]

    print()
    print(sum(train_) / len(train_))
    print(sum(dev_) / len(dev_))
    print(sum(test_) / len(test_))

def train(train_data):
    count_pos = 0
    count_neg = 0
    count_null = 0
    i = 0

    for batch in train_data:
        count_pos += torch.sum(batch.label.data == 2)
        count_neg += torch.sum(batch.label.data == 0)
        count_null += torch.sum(batch.label.data == 1)
        i += len(batch.label.data)

    print(count_pos + count_neg + count_null)

    return count_pos, count_neg, count_null


def evaluate(dataset, ratios, is_train=False):
    raw_accs = 0
    predictions = []
    truths = []

    total_true = [0, 0, 0]
    total_pred = [0, 0, 0]
    total_correct = [0, 0, 0]
    num_correct = 0
    num_examples = 0
    for batch in dataset:
        classifications = torch.LongTensor([generate_classifications(ratios) for i in range(0, len(batch.label.data))])
        if not is_train:
            print(ratios)
            evaluate_randomness(classifications)
        raw_accs += torch.sum(batch.label.data == classifications)

        for i in range(0, 3):
            total_true[i] += torch.sum(batch.label.data == i)
            total_pred[i] += torch.sum(classifications == i)
            total_correct[i] += torch.sum((classifications == i) * (batch.label.data == i))

            num_correct += float(torch.sum(classifications == batch.label.data))
            num_examples += len(batch.label.data)

        predictions.extend(classifications.numpy())
        truths.extend(batch.label.data.numpy())

    precision = [0, 0, 0]
    recall = [0, 0, 0]
    f1 = [0, 0, 0]
    for i in range(0, 3):
        if total_pred[i] == 0:
            precision[i] = 0.0
        else:
            precision[i] = total_correct[i] / total_pred[i]
        recall[i] = total_correct[i] / total_true[i]
        if precision[i] + recall[i] == 0:
            f1[i] = 0.0
        else:
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

    score = f1_score(list(predictions), list(truths), labels=[0, 1, 2], average=None)

    ratio_accs = raw_accs / num_examples
    print(str(score) + " " + str(f1))
    assert (score == f1).all()

    return ratio_accs, f1, precision, recall


def generate_classifications(ratios):
    num = random.random()
    if num < ratios[0]:
        return 0
    elif num < ratios[0] + ratios[1]:
        return 1
    return 2


def evaluate_randomness(classifications):
    counts = [0, 0, 0]
    ratio_counts = [0, 0, 0]
    for classification in classifications:
        counts[classification] += 1
    for i in range(0, 3):
        ratio_counts[i] = counts[i] / sum(counts)
    print("    " + str(ratio_counts))

if __name__ == '__main__':
    main()