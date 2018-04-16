import matplotlib.pyplot as plt
from matplotlib import ticker


train_accs = [[0.0, 0.3267372296364473, 0.0], [0.6190657045195594, 0.0, 0.0], [0.6190657045195594, 0.0, 0.0], [0.6190657045195594, 0.0, 0.0], [0.6190657045195594, 0.0, 0.0], [0.6210847975553858, 0.0, 0.03619909502262444], [0.6234974796432726, 0.0, 0.11396011396011395], [0.577831617201696, 0.0, 0.5717791411042944], [0.6190657045195594, 0.0, 0.0], [0.6050420168067226, 0.0, 0.5628342245989304], [0.6190657045195594, 0.0, 0.0], [0.6548596112311015, 0.0, 0.43685300207039335], [0.43966179861644883, 0.0, 0.5575757575757575], [0.6586517818806356, 0.0, 0.4411764705882353], [0.6645569620253164, 0.0, 0.510757717492984], [0.6351192804067266, 0.0, 0.18232044198895028], [0.6538461538461539, 0.0, 0.5634477254588987], [0.6568193008370261, 0.0, 0.5615999999999999], [0.6460409019402203, 0.0, 0.5851528384279476], [0.6449893390191898, 0.0, 0.5893238434163701], [0.5716096324461343, 0.0, 0.5895478567234292]]

dev_accs = [[0.0, 0.934850431271793, 0.0], [0.07830126078301261, 0.0, 0.008230452674897118], [0.07579787234042552, 0.0, 0.01606425702811245], [0.07579787234042552, 0.0, 0.01606425702811245], [0.07584830339321358, 0.0, 0.031872509960159355], [0.07777777777777778, 0.0, 0.10771992818671454], [0.07585644371941272, 0.0, 0.12919254658385093], [0.1097770154373928, 0.0, 0.1836441893830703], [0.07653575025176235, 0.0, 0.07194244604316546], [0.11687363038714389, 0.0, 0.18961864406779663], [0.07927677329624479, 0.0, 0.06299212598425197], [0.09509946627850557, 0.0, 0.16387959866220736], [0.13186813186813187, 0.0, 0.15572519083969466], [0.09261083743842365, 0.0, 0.15810920945395276], [0.10443864229765012, 0.0, 0.15797317436661698], [0.0973111395646607, 0.0, 0.1400437636761488], [0.12923864363403711, 0.0, 0.16056670602125148], [0.13127413127413126, 0.0, 0.15267175572519084], [0.13107822410147993, 0.0, 0.1458106637649619], [0.1265993265993266, 0.0, 0.14221218961625284], [0.14443500424808836, 0.0, 0.1451923076923077]]


def graph_attention(model, word_to_ix, ix_to_word, batch, using_GPU):
    (words, lengths), polarity, holder_target, label = batch.text, batch.polarity, batch.holder_target, batch.label
    log_probs, attention = model(words, polarity, holder_target, lengths)
    input_sentence = decode(words[:, 0], ix_to_word)
    print(str(log_probs.data.max(1)[1]) + str(label))
    print(input_sentence)
    instance_attention = attention[:, 0].data.numpy()
    print(instance_attention)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(instance_attention, cmap='bone')
    fig.colorbar(cax)
    # Set up axes
    ax.set_yticklabels([''] + input_sentence)
    ax.set_xticklabels([''])

    # Show label at every tick
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


# plot across each epoch
def plot(train_accs, dev_accs, test_accs):
    plt.plot(range(0, epochs + 1), dev_accs, c='red', label='Dev Set Accuracy')
    plt.plot(range(0, epochs + 1), train_accs, c='blue', label='Train Set Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. # of Epochs')
    plt.legend()
    plt.show()
