import json

datasets = ["acl_dev_eval.json", "acl_test.json", "acl_mpqa_eval.json",
            "acl_dev_tune.json", "C_train.json"]
datasets2 = ["acl_dev_eval_new.json", "acl_test_new.json", "mpqa_new.json"]  # "F_train.json", "C_train.json",

label_map = {"Positive": 2, "Negative": 0, "None": 1, "Null": 1}

for i in range(len(datasets)):
    set = datasets[i]
    print(set)

    doc_to_entity = {}
    labels = [0, 0, 0]
    labels_new = [{}, {}, {}]
    with open("../data/stanford_label_sent_boundary/" + set, "r", encoding="latin1") as rf:
        for line in rf:
            annot = json.loads(line)
            annot["label"] = label_map[annot["label"]]
            labels[annot["label"]] += 1
            labels_new[annot["label"]][str(annot["holder"]) + "; " + str(annot["target"])] = 1
            if annot["docid"] not in doc_to_entity:
                doc_to_entity[annot["docid"]] = {}
            if annot["holder"] not in doc_to_entity[annot["docid"]]:
                doc_to_entity[annot["docid"]][annot["holder"]] = 1
            if annot["target"] not in doc_to_entity[annot["docid"]]:
                doc_to_entity[annot["docid"]][annot["target"]] = 1

    '''
    with open("../data/new_annot/polarity_label/" + datasets2[i], "r", encoding="latin1") as rf:
        for line in rf:
            annot = json.loads(line)
            if annot["label"] != 1:
                if (str(annot["holder"]) + "; " + str(annot["target"])) not in labels_new[int(annot["label"])]:
                    print(annot["docid"])
                    print(str(annot["holder"]) + "; " + annot["target"] + " " + str(annot["label"]))
    '''

    print("    label breakdown = " + str(labels))
    num_docs = len(doc_to_entity)
    print("    # documents = " + str(num_docs))
    total_ents_all_docs = 0
    for doc in doc_to_entity:
        total_ents_all_docs += len(doc_to_entity[doc])
    print("    entities / document = " + str(total_ents_all_docs / num_docs))
