import json
import random

filename = "acl_dev_eval_new.json"
doc_to_sentences = {}
doc_to_sentences_co_occur = {}
all_data = []

with open("./data/new_annot/polarity_label_holdtarg/" + filename, "r", encoding="latin1") as cf:
    for line in cf:
        annot = json.loads(line)
        if annot["docid"] not in doc_to_sentences:
            sentences_co_occur = 0
            sentences = 0
            sentence_has_holder = False
            sentence_has_target = False
            for i in range(0, len(annot["holder_target"])):
                if annot["holder_target"][i] == "holder":
                    sentence_has_holder = True
                if annot["holder_target"][i] == "target":
                    sentence_has_target = True
                if annot["token"][i] == ".":
                    sentences += 1
                    if sentence_has_holder and sentence_has_target:
                        sentences_co_occur += 1
                    sentence_has_holder = False
                    sentence_has_target = False
            doc_to_sentences[annot["docid"]] = sentences
            doc_to_sentences_co_occur[annot["docid"]] = sentences_co_occur
        #  print(doc_to_sentences[annot["docid"]])
        annot["co_occurrences"] = str(doc_to_sentences_co_occur[annot["docid"]])
        all_data.append(annot)
print(doc_to_sentences)
print(doc_to_sentences_co_occur)

doc_to_sentences = {}
doc_to_sentences_co_occur = {}

with open("./data/new_annot/feature/" + filename, "w", encoding="latin1") as wf:
    for annot in all_data:
        json.dump(annot, wf)
        wf.write("\n")

