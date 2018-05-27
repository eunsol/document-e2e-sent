import json

type_prior_polarity_to_polarity = {('strongsubj', 'negative'):"strongneg",
                                  ('weaksubj', 'negative'):"weakneg",
                                  ('weaksubj', 'neutral'):"neutral",
                                  ('strongsubj', 'neutral'):"neutral",
                                  ('weaksubj', 'positive'):"weakpos",
                                  ('strongsubj', 'positive'):"strongpos"
                                   }

word_to_polarity = {}
with open("./resources/subjclueslen1-HLTEMNLP05.tff", 'r') as f:
    i = 0
    for line in f:
        word_line = line.split()
        word = (word_line[2].split("="))[1]
        type_ = (word_line[0].split("="))[1]
        prior_polarity = (word_line[5].split("="))[1]
        # ignore "both"
        if prior_polarity == "both":
            continue
        polarity = type_prior_polarity_to_polarity[(type_, prior_polarity)]
        word_to_polarity[word] = polarity


filenames = []# ["acl_dev_eval.json", "acl_dev_tune.json", "acl_mpqa_eval.json", "acl_test.json"]
for filename in filenames:
    all_data = []
    print(filename)
    with open("./data/stanford_label_sent_boundary_with_label/" + filename, "r", encoding="latin1") as af:
        for line in af:
            annot = json.loads(line)
            polarity = []
            for token in annot["token"]:
                if token in word_to_polarity.keys():
                    polarity.append(word_to_polarity[token])
                else:
                    polarity.append("<unk>")
            annot["polarity"] = polarity
            all_data.append(annot)

    with open("./data/" + filename, "w", encoding="latin1") as wf:
        for line in all_data:
            json.dump(line, wf)
            wf.write('\n')
    '''
	print("evaluating validity...")
	# testing
	with open("./data/new_annot/polarity/" + filename, "r", encoding="latin1") as af:
	    for line in af:
		annot = json.loads(line)
		polarities = annot["polarity"]
		tokens = annot["token"]
		if len(polarities) != len(tokens):
		    print("length problem: " + str(annot))
		for i in range(0, len(polarities)):
		    if polarities[i] == "<unk>":
			if tokens[i] in word_to_polarity.keys():
			    print("something not labelled")
		    else:
			if word_to_polarity[tokens[i]] != polarities[i]:
			    print("wrong token")
    '''
