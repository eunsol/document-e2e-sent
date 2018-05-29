import json

filenames = ["C_train.json", "F_train.json", "acl_dev_eval.json", "acl_test.json", "acl_mpqa_eval.json"]
for filename in filenames:
    print(filename)
    all_data = []
    with open("./data/final/" + filename, "r", encoding="latin1") as af:
        for line in af:
            annot = json.loads(line)
            holder_target = []
            tokens = annot["token"]
            holder_target = ["none" for i in range(0, len(tokens))]
            for holder in annot["holder_index"]:
                for i in range(holder[0], holder[1] + 1):
                    holder_target[i] = "holder"
            for target in annot["target_index"]:
                for i in range(target[0], target[1] + 1):
                    holder_target[i] = "target"
            annot["holder_target"] = holder_target
            all_data.append(annot)

    with open("./data/final/" + filename, "w", encoding="latin1") as wf:
        for line in all_data:
            json.dump(line, wf)
            wf.write('\n')

    print("evaluating validity...")
    # testing
    with open("./data/final/" + filename, "r", encoding="latin1") as ef:
        for line in ef:
            annot = json.loads(line)
            holder_target = annot["holder_target"]
            for i in range(0, len(holder_target)):
                if holder_target[i] == "holder":
                    found_range = False
                    for holder in annot["holder_index"]:
                        if i >= holder[0] and i <= holder[1]:
                            found_range = True
                    if not found_range:
                        print("holder problem")
                if holder_target[i] == "target":
                    found_range = False
                    for target in annot["target_index"]:
                        if i >= target[0] and i <= target[1]:
                            found_range = True
                    if not found_range:
                        print("target problem")
