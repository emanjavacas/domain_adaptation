
import random
import matplotlib.pyplot as plt
import numpy as np
import cPickle as p
import json
from datetime import datetime
from collections import defaultdict


def take(g, n):
    c = 0
    while c < n:
        yield g.next()
        c += 1


def shuffle_seq(seq, seed=448):
    return sorted(seq, key=lambda k: random.random())


def plot_conf_matrix(cm, labels, title='Confusion Matrix', cmap=plt.cm.OrRd):
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    n_ticks = np.arange(len(labels))
    plt.xticks(n_ticks, labels, fontsize=8, rotation=45)
    plt.yticks(n_ticks, labels, fontsize=8)
    plt.tight_layout()
    plt.xlabel('Predicted label', fontsize=14)
    plt.ylabel('True label', fontsize=14)


def normalize_cm(cm):
    total_rows = np.asarray(sum(np.transpose(cm)), dtype=np.float)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm / total_rows
        cm_norm[cm_norm == np.inf] = 0
        cm_norm = np.nan_to_num(cm_norm)
        return cm_norm


def per_tag_scores(y_true, y_pred):
    scores = defaultdict(lambda: defaultdict(int))
    assert len(y_true) == len(y_pred)
    for i in range(len(y_true)):
        true_tag = y_true[i]
        guessed_tag = y_pred[i]
        if true_tag == guessed_tag:
            scores[true_tag]['tp'] += 1
        else:
            scores[true_tag]['fn'] += 1
            scores[guessed_tag]['fp'] += 1
    return dict(scores)


def recall(tp=0, fn=0, fp=0):
    return np.float64(tp) / (tp + fn)


def precision(tp=0, fn=0, fp=0):
    return np.float64(tp) / (tp + fp)


def f1(tp=0, fn=0, fp=0):
    den = 2 * tp
    num = den + fp + fn
    return np.float64(den) / num


def sorted_scores(scores, fn=precision):
    """
    sorted scores per tag by tag frequency.
    scores = per_tag_scores(y_true, y_pred)
    sorted_scores(scores)
    >>> [(u'N', 0.91368227731864093),
         (u'P', 0.97044804575786459),
         (u',', 0.9981412639405205),
         (u'PRO', 0.98971193415637859),
         ...]
    """
    return sorted([(k, fn(**v)) for k, v in scores.items()],
                  key=lambda x: sum(scores[x[0]].values()), reverse=True)


def pickle_this(fname, obj):
    fname += '_' + datetime.now().isoformat() + '.pickle'
    with open(fname, 'w') as f:
        p.dump(obj, f)

        
def unpickle_this(fname):
    with open(fname, 'r') as f:
        return p.load(f)

    
def serialize_results(fname, y_test, y_pred, labels):
    result = {"y_test": y_test, "y_pred": y_pred, "labels": labels}
    with codecs.open(fname, "w+", "utf-8") as f:
        json.dump(result, f)
