import json

confirm_filepath = "./data/stanford_label_sent_boundary/"
confirm_files = ["acl_dev_eval.json", "acl_dev_tune.json", "acl_test.json", "acl_mpqa_eval.json"]  #, "train_all.json"]

'''
for i in range(len(confirm_files)):
    confirm_file = confirm_files[i]
    print(confirm_file)

    all_data = []
    with open(confirm_filepath + confirm_file, "r", encoding="latin1") as cf:
        for line in cf:
            annot = json.loads(line)

            has_holder = [False for i in range(len(annot["sent_boundaries"]))]
            sent_ind = 0
            for holder_bounds in sorted(annot["holder_index"], key=lambda x: x[0]):
                for ind in holder_bounds:
                    while sent_ind < len(annot["sent_boundaries"]) - 1 and ind >= annot["sent_boundaries"][sent_ind + 1]:
                        sent_ind += 1
                    has_holder[sent_ind] = True
            has_target = [False for i in range(len(annot["sent_boundaries"]))]
            sent_ind = 0
            for target_bounds in sorted(annot["target_index"], key=lambda x: x[0]):
                for ind in target_bounds:
                    while sent_ind < len(annot["sent_boundaries"]) - 1 and ind >= annot["sent_boundaries"][sent_ind + 1]:
                        sent_ind += 1
                    has_target[sent_ind] = True
            has_both = [has_holder[i] and has_target[i] for i in range(len(has_holder))]
            count_has_both = sum(has_both)

            # Update co-occurrences based on new sentence boundaries
            annot["co_occurrences"] = count_has_both

            # Get doc ht classification based on sentence features
            classify = 1
            counts = [0, 0, 0]
            for i in range(len(has_both)):
                if has_both[i]:
                    if annot["sent_seq"][i] > 2:
                        counts[2] += 1
                    elif annot["sent_seq"][i] == 2:
                        counts[1] += 1
                    else:
                        counts[0] += 1
            # At least 1 positive = classify as positive
            if counts[2] >= 1:
                classify = 2
            elif counts[0] >= 1:  # At least 1 negative & no positive = classify as negative
                classify = 0
            else:  # No negative or positive (all neutral or no co-occurs)
                classify = 1
            annot["classify"] = classify
            
            if annot["label"] == "Positive":
                annot["label"] = 2
            elif annot["label"] == "Negative":
                annot["label"] = 0
            else:
                annot["label"] = 1

            assert counts[0] + counts[1] + counts[2] == count_has_both

            all_data.append(annot)

    print(len(all_data))

    with open("./data/stanford_label_sent_boundary_with_label/" + confirm_file, "w", encoding="latin1") as cf:
        for line in all_data:
            json.dump(line, cf)
            cf.write("\n")

'''
for fn in confirm_files:
    print(fn)
    # Compare with results from just sentence
    all_correct_data = [[], [], []]
    all_wrong_data_pred = [[], [], []]
    all_wrong_data_acts = [[], [], []]

    num_correct_data = [0, 0, 0]
    num_wrong_data_pred = [0, 0, 0]
    num_wrong_data_acts = [0, 0, 0]
    with open("./data/stanford_label_sent_boundary_with_label/" + fn, "r", encoding="latin1") as rf:
        for line in rf:
            annot = json.loads(line)
            label = annot["label"]
            pred = annot["classify"]
            if label == pred:
                # all_correct_data[label].append(annot)
                num_correct_data[label] += 1
            # all_wrong_data_pred[pred].append(annot)
            # all_wrong_data_acts[label].append(annot)
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
        if num_wrong_data_pred[i] == 0:
            precision[i] = 0
        else:
            precision[i] = num_correct_data[i] / num_wrong_data_pred[i]
        recall[i] = num_correct_data[i] / num_wrong_data_acts[i]
        if precision[i] + recall[i] == 0:
            total_f1[i] = 0
        else:
            total_f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
    print(precision)
    print(recall)
    print(total_f1)
# '''