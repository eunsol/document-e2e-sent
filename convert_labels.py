import json

all_data = []
filename = ".json"
with open("./data/new_annot/polarity/" + filename, "r", encoding="latin1") as af:
    for line in af:
        annot = json.loads(line)
        polarity = []
        if annot["label"] == "Positive":
            annot["label"] = 2
        elif annot["label"] == "Negative":
            annot["label"] = 0
        else:
            annot["label"] = 1
        all_data.append(annot)

with open("./data/new_annot/polarity_label/" + filename, "w", encoding="latin1") as wf:
    for line in all_data:
        json.dump(line, wf)
        wf.write('\n')

'''
print("evaluating validity...")
# testing
with open("./data/new_annot/polarity_label/" + filename, "r", encoding="latin1") as ef:
    i = 0
    for line in ef:
        annot = json.loads(line)
        print(line == all_data[i])
        if annot["label"] == 1 and all_data[i]["label"] != "Positive":
            print("incorrectly labelled 1:" + str(line))
        elif annot["label"] == -1 and all_data[i]["label"] != "Negative":
            print("incorrectly labelled -1:" + str(line))
        elif annot["label"] == 0:
            print("incorrectly labelled 0:" + str(line))
'''
