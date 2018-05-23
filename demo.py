import json


def make_key(holder_inds, target_inds):
    key = ""
    key += str(holder_inds) + "; "
    key += str(target_inds)
    return key


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


def convert_to_list(str):
    str = str[1:len(str) - 1]
    list = []
    element = ""
    for char in str:
        if char == ",":
            list.append(element)
            element = ""
        elif char != " ":
            element += char
    list.append(element)
    return list


docs_to_data = {}
docs_to_num_right_wrong = {}

with open("./error_analysis/final/right_docs_dev.json", "r", encoding="latin1") as rf:
    for line in rf:
        annot = json.loads(line)
        if annot["docid"] not in docs_to_data:
            docs_to_data[annot["docid"]] = []
            docs_to_num_right_wrong[annot["docid"]] = {"right": 0, "wrong": 0}
        docs_to_data[annot["docid"]].append(annot)
        if annot["actual"] != 1:
            docs_to_num_right_wrong[annot["docid"]]["right"] += 1

with open("./error_analysis/final/wrong_docs_dev.json", "r", encoding="latin1") as rf:
    for line in rf:
        annot = json.loads(line)
        if annot["docid"] not in docs_to_data:
            docs_to_data[annot["docid"]] = []
            docs_to_num_right_wrong[annot["docid"]] = {"right": 0, "wrong": 0}
        docs_to_data[annot["docid"]].append(annot)
        if annot["actual"] != 1:
            docs_to_num_right_wrong[annot["docid"]]["wrong"] += 1

doc_to_ht_to_preds = {}
doc_to_ht_to_acts = {}

for doc in docs_to_data:
    annots = docs_to_data[doc]
    for annot in annots:
        ht_key = make_key(annot["holders"], annot["targets"])
        probs = str(annot["probabilities"])
        doc_to_ht_to_preds[ht_key] = probs
        doc_to_ht_to_acts[ht_key] = annot["actual"]

docs_to_use = []
for doc in sorted(docs_to_num_right_wrong.items(), key=lambda x: x[1]["wrong"], reverse=False):
    docs_to_use.append(doc)
    if len(docs_to_use) > 30:
        break

doc_to_ht_to_annot = {}
doc_to_entity = {}
with open("./data/new_annot/none/acl_dev_eval_new.json", "r", encoding="latin1") as rf:
    for line in rf:
        annot = json.loads(line)
        docid = annot["docid"]
        if docid not in doc_to_ht_to_annot:
            doc_to_ht_to_annot[docid] = {}
            doc_to_entity[docid] = {}
        ht_key = make_key(annot["holder_index"], annot["target_index"])
        if ht_key not in doc_to_ht_to_annot[docid]:
            doc_to_ht_to_annot[docid][ht_key] = annot

        holder = str(annot["holder"])
        if holder not in doc_to_entity[docid]:
            doc_to_entity[docid][holder] = annot["holder_index"]
        target = str(annot["target"])
        if target not in doc_to_entity[docid]:
            doc_to_entity[docid][target] = annot["target_index"]

text_to_docid = {}
text_to_ht_to_pred = {}
text_to_ht_to_acts = {}
text_to_entity = {}
texts_ht_preds = []
for doc in docs_to_use:
    docid = doc[0]
    ht_to_annot = doc_to_ht_to_annot[docid]
    for ht in ht_to_annot:
        pred = doc_to_ht_to_preds[ht]
        act = doc_to_ht_to_acts[ht]
        token = doc_to_ht_to_annot[docid][ht]["token"]
        if docid == "AFP_ENG_20101123.0339":
            for entity in doc_to_entity[docid]:
                for indices in doc_to_entity[docid][entity]:
                    token[indices[0]] = "{" + token[indices[0]]
                    token[indices[1]] = token[indices[0]] + "}_" + str(entity)
        holder = doc_to_ht_to_annot[docid][ht]["holder"]
        target = doc_to_ht_to_annot[docid][ht]["target"]
        text = convert_to_str(token)
        if text not in text_to_ht_to_pred:
            text_to_ht_to_pred[text] = {}
            text_to_ht_to_acts[text] = {}
            text_to_entity[text] = doc_to_entity[docid]
            text_to_docid[text] = docid
        text_to_ht_to_pred[text]["(\"" + str(holder) + "\", \"" + str(target) + "\")"] = pred
        text_to_ht_to_acts[text]["(\"" + str(holder) + "\", \"" + str(target) + "\")"] = act


# AFP_ENG_20101123.0339
# AFP_ENG_20100322.0387
# APW_ENG_20090729.0699
# XIN_ENG_20100125.0144
# XIN_ENG_20090902.0333
def main():
    print(docs_to_use)
    print()
    for text in text_to_ht_to_pred:
        print(text_to_docid[text])
        print(text)
        print(text_to_entity[text])
        ht_to_pred = text_to_ht_to_pred[text]
        for ht in ht_to_pred:
            pred_label = 0
            prob_list = convert_to_list(ht_to_pred[ht])
            for i in range(3):
                if float(prob_list[i]) > float(prob_list[pred_label]):
                    pred_label = i
            act_label = text_to_ht_to_acts[text][ht]
            if pred_label != 1:
                print("    " + str(ht) + ", ")  # + ": " + str(prob_list) + " " + str(pred_label) + " " + str(act_label))

        print()


if __name__ == "__main__":
    main()
