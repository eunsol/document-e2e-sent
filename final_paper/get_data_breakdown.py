import json

datasets = ["F_train.json", "C_train.json", "acl_dev_eval.json", "acl_test.json", "acl_mpqa_eval.json"]

for set in datasets:
    print(set)
    doc_to_entity = {}
    labels = [0, 0, 0]
    with open("../data/final/" + set, "r", encoding="latin1") as rf:
        for line in rf:
            annot = json.loads(line)
            labels[annot["label"]] += 1
            if annot["docid"] not in doc_to_entity:
                doc_to_entity[annot["docid"]] = {}
            if annot["holder"] not in doc_to_entity[annot["docid"]]:
                doc_to_entity[annot["docid"]][annot["holder"]] = 1
            if annot["target"] not in doc_to_entity[annot["docid"]]:
                doc_to_entity[annot["docid"]][annot["target"]] = 1
    print("    label breakdown = " + str(labels))
    num_docs = len(doc_to_entity)
    print("    # documents = " + str(num_docs))
    total_ents_all_docs = 0
    for doc in doc_to_entity:
        total_ents_all_docs += len(doc_to_entity[doc])
    print("    entities / document = " + str(total_ents_all_docs / num_docs))
