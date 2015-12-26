
import random
import matplotlib.pyplot as plt
import numpy as np
import cPickle as p
from datetime import datetime


def take(g, n):
    c = 0
    while c < n:
        yield g.next()
        c += 1


def shuffle(seq, seed=448):
    return sorted(seq, key=lambda k: random.random())


def plot_conf_matrix(cm, labels, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    n_ticks = np.arange(len(labels))
    plt.xticks(n_ticks, labels, fontsize=8, rotation=45)
    plt.yticks(n_ticks, labels, fontsize=8)
    plt.tight_layout()
    plt.xlabel('Predicted label', fontsize=14)
    plt.ylabel('True label', fontsize=14)


def pickle_this(fname, obj):
    fname += '_' + datetime.now().isoformat() + '.pickle'
    with open(fname, 'w') as f:
        p.dump(obj, f)
