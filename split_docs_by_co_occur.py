import json

confirm_filepath = "./data/stanford_label_sent_boundary/"
read_filepath = "./data/new_annot/feature/"
write_filepath_yes = "./data/has_co_occurs/"
write_filepath_no = "./data/no_co_occurs/"
read_files = ["new_train.json", "acl_dev_eval_new.json", "acl_dev_tune_new.json", "acl_test_new.json", "mpqa_new.json"]
confirm_files = ["train_all.json", "acl_dev_eval.json", "acl_dev_tune.json", "acl_test.json", "acl_mpqa_eval.json"]
write_files = ["new_train.json", "acl_dev_eval_new.json", "acl_dev_tune_new.json", "acl_test_new.json", "mpqa_new.json"]

read_combine_files = ["E_train.json", "F_train.json", "G_train.json", "H_train.json"]
combine_files = ["new_train.json", "acl_dev_tune_new.json"]
write_combine_files = ["E_train.json", "F_train.json", "G_train.json", "H_train.json"]


def make_key(holder_inds, target_inds):
    key = ""
    key += str(holder_inds) + "; "
    key += str(target_inds)
    return key


for i in range(len(read_files)):
    read_file = read_files[i]
    confirm_file = confirm_files[i]
    write_file = write_files[i]
    print(read_file)

    # Getting sentence boundaries
    doc_to_sentbounds = {}
    doc_to_sentseq = {}
    with open(confirm_filepath + confirm_file, "r", encoding="latin1") as cf:
        for line in cf:
            annot = json.loads(line)
            if annot["docid"] not in doc_to_sentbounds:
                doc_to_sentbounds[annot["docid"]] = annot["sent_boundaries"]
                doc_to_sentseq[annot["docid"]] = annot["sent_seq"]

    all_data = []
    doc_to_ht_to_co_occurs = {}
    doc_to_names_to_co_occurs = {}

    # Getting co-occurring & non-co-occurring pairs
    # Adding sentence boundaries
    # Adding sentence classifications
    co_occur_exs = []
    no_co_occur_exs = []
    with open(read_filepath + read_file, "r", encoding="latin1") as rf:
        for line in rf:
            annot = json.loads(line)
            # Getting co-occurrences
            annot["co_occurrences"] = int(annot["co_occurrences"])
            if annot["docid"] not in doc_to_ht_to_co_occurs:
                doc_to_ht_to_co_occurs[annot["docid"]] = {}
                doc_to_names_to_co_occurs[annot["docid"]] = {}
            ht_key = make_key(annot["holder_index"], annot["target_index"])
            ht_name_key = str(annot["holder_index"]) + "; " + str(annot["target_index"])
            doc_to_ht_to_co_occurs[annot["docid"]][ht_key] = annot["co_occurrences"]
            doc_to_names_to_co_occurs[annot["docid"]][ht_name_key] = annot["co_occurrences"]

            # Adding sentence boundaries
            annot["sent_boundaries"] = doc_to_sentbounds[annot["docid"]]
            # Adding sentence classifications
            annot["sent_seq"] = doc_to_sentseq[annot["docid"]]
            all_data.append(annot)

    inconsistencies = 0
    num_not_found = 0
    # Confirming co-occurrence of pairs--many inconsistencies... okay then...
    # Iterate through all data and get co-occurrences
    # Also get sentence RNN classification
    for annot in all_data:
        # ht_key = make_key(annot["holder_index"], annot["target_index"])
        # ht_name_key = str(annot["holder_index"]) + "; " + str(annot["target_index"])
        co_occurs_prev = annot["co_occurrences"]

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
        if co_occurs_prev != count_has_both:
            inconsistencies += 1
            print(annot["sent_boundaries"])
            print("inconsistent: " + str(count_has_both) + " " + str(has_both) + " " + str(co_occurs_prev))
            print()

        # Update co-occurrences based on new sentence boundaries
        annot["co_occurrences"] = count_has_both

        # Get doc ht classification based on sentence features
        classify = 1
        counts = [0, 0, 0]
        for i in range(len(has_both)):
            if has_both[i]:
                classify = 0
                if annot["sent_seq"][i] > 3:
                    counts[2] += 1
                elif annot["sent_seq"][i] == 3:
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

        if annot["co_occurrences"] > 0:
            co_occur_exs.append(annot)
        else:
            no_co_occur_exs.append(annot)

    print(num_not_found)
    print(inconsistencies)
    print()
    print(len(all_data))
    print(len(co_occur_exs))
    print(len(no_co_occur_exs))
    assert len(all_data) == len(co_occur_exs) + len(no_co_occur_exs)

    # Write co-occurring examples to doc
    with open(write_filepath_yes + write_file, "w", encoding="latin1") as wf:
        for annot in co_occur_exs:
            json.dump(annot, wf)
            wf.write("\n")

    # Write non-co-occurring examples to doc
    with open(write_filepath_no + write_file, "w", encoding="latin1") as wf:
        for annot in no_co_occur_exs:
            json.dump(annot, wf)
            wf.write("\n")

# Create maps for combined training datasets from written-to files
doc_to_ht_to_annot = {}
doc_to_sentbounds = {}
doc_to_sentseq = {}
for combine_file in combine_files:
    print("    " + str(combine_file))
    with open(write_filepath_yes + combine_file, "r", encoding="latin1") as cf:
        for line in cf:
            annot = json.loads(line)
            if annot["docid"] not in doc_to_ht_to_annot:
                doc_to_ht_to_annot[annot["docid"]] = {}
                doc_to_sentbounds[annot["docid"]] = annot["sent_boundaries"]
                doc_to_sentseq[annot["docid"]] = annot["sent_seq"]
            ht_key = make_key(annot["holder_index"], annot["target_index"])
            doc_to_ht_to_annot[annot["docid"]][ht_key] = annot

    with open(write_filepath_no + combine_file, "r", encoding="latin1") as cf:
        for line in cf:
            annot = json.loads(line)
            if annot["docid"] not in doc_to_ht_to_annot:
                doc_to_ht_to_annot[annot["docid"]] = {}
                doc_to_sentbounds[annot["docid"]] = annot["sent_boundaries"]
                doc_to_sentseq[annot["docid"]] = annot["sent_seq"]
            ht_key = make_key(annot["holder_index"], annot["target_index"])
            doc_to_ht_to_annot[annot["docid"]][ht_key] = annot

print("    train_all.json")
with open(confirm_filepath + "train_all.json", "r", encoding="latin1") as cf:
    for line in cf:
        annot = json.loads(line)
        if annot["docid"] not in doc_to_sentbounds:
            doc_to_sentbounds[annot["docid"]] = annot["sent_boundaries"]
            doc_to_sentseq[annot["docid"]] = annot["sent_seq"]

# Get the combined training datasets
for i in range(len(read_combine_files)):
    read_file = read_combine_files[i]
    write_file = write_combine_files[i]
    print(read_file)

    all_data = []
    with open(read_filepath + read_file, "r", encoding="latin1") as rf:
        for line in rf:
            annot = json.loads(line)
            annot["sent_boundaries"] = doc_to_sentbounds[annot["docid"]]
            annot["sent_seq"] = doc_to_sentseq[annot["docid"]]
            all_data.append(annot)

    write_co_occur = []
    write_no_co_occur = []
    inconsistencies = 0
    num_not_found = 0
    # Confirming co-occurrence of pairs--many inconsistencies... okay then...
    # Iterate through all data and get co-occurrences
    # Also get sentence RNN classification
    for annot in all_data:
        # ht_key = make_key(annot["holder_index"], annot["target_index"])
        # ht_name_key = str(annot["holder_index"]) + "; " + str(annot["target_index"])
        co_occurs_prev = annot["co_occurrences"]

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
        if int(co_occurs_prev) != int(count_has_both):
            inconsistencies += 1
            print(annot["holder_index"])
            print(annot["target_index"])
            print(annot["sent_seq"])
            print(annot["sent_boundaries"])
            print("inconsistent: " + str(count_has_both) + " " + str(has_both) + " " + str(co_occurs_prev))
            print()

        # Update co-occurrences based on new sentence boundaries
        annot["co_occurrences"] = count_has_both

        # Get doc ht classification based on sentence features
        classify = 1
        counts = [0, 0, 0]
        for i in range(len(has_both)):
            if has_both[i]:
                classify = 0
                if annot["sent_seq"][i] > 3:
                    counts[2] += 1
                elif annot["sent_seq"][i] == 3:
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

        if annot["co_occurrences"] > 0:
            write_co_occur.append(annot)
        else:
            write_no_co_occur.append(annot)

        # Check if existed previously, and check against the existent label if it does
        ht_key = make_key(annot["holder_index"], annot["target_index"])
        if annot["docid"] in doc_to_ht_to_annot and ht_key in doc_to_ht_to_annot[annot["docid"]]:
            assert doc_to_ht_to_annot[annot["docid"]][ht_key].keys() == annot.keys()
            assert doc_to_ht_to_annot[annot["docid"]][ht_key]["sent_boundaries"] == annot["sent_boundaries"]
            assert doc_to_ht_to_annot[annot["docid"]][ht_key]["sent_seq"] == annot["sent_seq"]
            assert annot["co_occurrences"] == doc_to_ht_to_annot[annot["docid"]][ht_key]["co_occurrences"]
            assert annot["classify"] == doc_to_ht_to_annot[annot["docid"]][ht_key]["classify"]
            print(annot["label"])
            print(doc_to_ht_to_annot[annot["docid"]][ht_key]["label"])
            if annot["label"] != doc_to_ht_to_annot[annot["docid"]][ht_key]["label"]:
                if annot["label"] == 1:
                    annot["label"] = doc_to_ht_to_annot[annot["docid"]][ht_key]["label"]
            annot["token"] = doc_to_ht_to_annot[annot["docid"]][ht_key]["token"]  # apparently these tokens are right...
            annot["holder_rank"] = doc_to_ht_to_annot[annot["docid"]][ht_key]["holder_rank"]
            annot["target_rank"] = doc_to_ht_to_annot[annot["docid"]][ht_key]["target_rank"]
            annot["holder"] = doc_to_ht_to_annot[annot["docid"]][ht_key]["holder"]
            annot["target"] = doc_to_ht_to_annot[annot["docid"]][ht_key]["target"]
            assert annot["holder_target"] == doc_to_ht_to_annot[annot["docid"]][ht_key]["holder_target"]
            assert annot["polarity"] == doc_to_ht_to_annot[annot["docid"]][ht_key]["polarity"]
            # print(str(annot["docid"]) + " " + str(ht_key))
            assert doc_to_ht_to_annot[annot["docid"]][ht_key] == annot

    '''
    for annot in all_data:
        print(annot)
        ht_key = make_key(annot["holder_index"], annot["target_index"])
        write_annot = doc_to_ht_to_annot[annot["docid"]][ht_key]
        print(annot)
        print(write_annot)
        if write_annot["co_occurrences"] > 0:
            write_co_occur.append(write_annot)
        else:
            write_no_co_occur.append(write_annot)
    '''
    print(len(all_data))
    print(len(write_co_occur))
    print(len(write_no_co_occur))
    assert len(all_data) == len(write_co_occur) + len(write_no_co_occur)

    print("Writing...")
    # Write co-occurring examples to doc
    with open(write_filepath_yes + write_file, "w", encoding="latin1") as wf:
        for annot in write_co_occur:
            json.dump(annot, wf)
            wf.write("\n")

    # Write non-co-occurring examples to doc
    with open(write_filepath_no + write_file, "w", encoding="latin1") as wf:
        for annot in write_no_co_occur:
            json.dump(annot, wf)
            wf.write("\n")
