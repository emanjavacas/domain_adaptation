
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix
from utils import deserialize_results, intersection


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


        
def plot_metric_over_years(results, years, metric, tags, title="My title"):
    assert len(results) == len(years)
    for result, year in zip(results, years):
        for y_true, y_pred, _ in result:
            scores = per_tag_scores(y_true, y_pred)
            scores = {k: metric(**scores[k]) for k in tags}
            plt.scatter(scores.keys(), scores.values())

            
# y_true, y_pred, labels = deserialize_results("models/1500.json")
# cm = confusion_matrix(y_true, y_pred)
# plot_conf_matrix(cm, labels= labels)

# def counts_per_tag(sents):
#     return Counter(tag for sent in sents for word, tag in sent)

# from penn_data import pos_from_range
# counts = dict()
# test = (1400, 1500, 2000)
# test_counts = counts_per_tag(pos_from_range(*test))
# for start in range(1400, 1850, 50):
#     train = (start, start + 100, 10000)
#     sents = pos_from_range(*train)
#     train_counts = counts_per_tag(sents)
#     counts[start] = train_counts

# todo: measure correlation between per tag accuracy and counts over time
