import json
import random

files = ["new_train.json", "F_train.json", "acl_dev_eval_new.json", "acl_dev_tune_new.json", "acl_test_new.json", "mpqa_new.json"]

for filename in files:
    print(filename)
    '''
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
    '''

    all_data = []
    doc_to_entity_to_num_mentions = {}

    with open("./data/new_annot/feature/" + filename, "r", encoding="latin1") as cf:
        for line in cf:
            annot = json.loads(line)
            all_data.append(annot)
            if annot["docid"] not in doc_to_entity_to_num_mentions:
                doc_to_entity_to_num_mentions[annot["docid"]] = {}
            if annot["holder"] not in doc_to_entity_to_num_mentions[annot["docid"]]:
                doc_to_entity_to_num_mentions[annot["docid"]][annot["holder"]] = len(annot["holder_index"])
            else:
                assert doc_to_entity_to_num_mentions[annot["docid"]][annot["holder"]] == len(annot["holder_index"])
            if annot["target"] not in doc_to_entity_to_num_mentions[annot["docid"]]:
                doc_to_entity_to_num_mentions[annot["docid"]][annot["target"]] = len(annot["target_index"])
            else:
                assert doc_to_entity_to_num_mentions[annot["docid"]][annot["target"]] == len(annot["target_index"])

    doc_to_num_mentions_to_entity = {}

    # Sorting
    for doc in doc_to_entity_to_num_mentions:
        entity_to_mentions = doc_to_entity_to_num_mentions[doc]
        if doc not in doc_to_num_mentions_to_entity:
            doc_to_num_mentions_to_entity[doc] = {}
        for entity in entity_to_mentions:
            if entity_to_mentions[entity] not in doc_to_num_mentions_to_entity[doc]:
                doc_to_num_mentions_to_entity[doc][entity_to_mentions[entity]] = []
            doc_to_num_mentions_to_entity[doc][entity_to_mentions[entity]].append(entity)

    doc_to_rank_to_entity = {}

    max_rank = 0
    rank_distr = {}
    # Ranking
    for doc in doc_to_entity_to_num_mentions:
        mentions_to_entity = doc_to_num_mentions_to_entity[doc]
        if doc not in doc_to_rank_to_entity:
            doc_to_rank_to_entity[doc] = {}
        rank = 0
        for mention in sorted(mentions_to_entity, reverse=True):
            if doc == "AFP_ENG_20101230.0350":
                print(mentions_to_entity[mention])
            rank += 1
            doc_to_rank_to_entity[doc][rank] = mentions_to_entity[mention]
        if rank not in rank_distr:
            rank_distr[rank] = 0
        rank_distr[rank] += 1
        if rank > max_rank:
            max_rank = rank

    print(max_rank)
    print(rank_distr)

    doc_to_entity_to_rank = {}
    # Reversing rank -> entity to entity -> rank
    for doc in doc_to_rank_to_entity:
        rank_to_entity = doc_to_rank_to_entity[doc]
        if doc not in doc_to_entity_to_rank:
            doc_to_entity_to_rank[doc] = {}
        for rank in doc_to_rank_to_entity[doc]:
            for entity in doc_to_rank_to_entity[doc][rank]:
                doc_to_entity_to_rank[doc][entity] = rank

    # Adding to feature to all_data
    for annot in all_data:
        annot["holder_rank"] = doc_to_entity_to_rank[annot["docid"]][annot["holder"]]
        annot["target_rank"] = doc_to_entity_to_rank[annot["docid"]][annot["target"]]

    with open("./data/new_annot/feature/" + filename, "w", encoding="latin1") as wf:
        for annot in all_data:
            json.dump(annot, wf)
            wf.write("\n")