
import json
import argparse
import codecs

from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix
from utils import deserialize_results, intersection, read_results


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
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.float64(tp) / (tp + fn)


def precision(tp=0, fn=0, fp=0):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.float64(tp) / (tp + fp)


def f1(tp=0, fn=0, fp=0):
    with np.errstate(divide='ignore', invalid='ignore'):
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
#    assert len(results) == len(years)
    per_year = defaultdict(list)
    tags = set(tags)
    for result, year in zip(results, years):
        y_true, y_pred, _ = result
        scores = per_tag_scores(y_true, y_pred)
        for k, v in scores.items():
            if k in tags:
                value = metric(**v)
                per_year[k] += [value]
    for tag, scores in per_year.items():
        plt.plot(range(len(scores)), scores, label=tag)
    plt.legend()

def plotlify_results(results, years, metric, tags):
    per_tag = defaultdict(dict)
    tags = set(tags)
    for result, year in zip(results, years):
        y_true, y_pred, _ = result
        scores = per_tag_scores(y_true, y_pred)
        for tag, v in scores.items():
            if tag in tags:
                per_tag[tag][year] = metric(**v)
    output = []
    for tag, d in per_tag.items():
        x, y = zip(*sorted(d.items()))
        output.append({
            "x": x,
            "y": y,
            "name": tag
        })
    return output

def to_plotly_json(results, years, metric, tags, fname):
    plotly_obj = plotlify_results(results, years, metric, tags)
    with codecs.open(fname, "w+", "utf-8") as f:
        json.dump(plotly_obj, f)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluation")
    parser.add_argument("files", nargs='+')
    parser.add_argument("-t", "--tags", default="most_freq")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-m", "--metric", required=True)
    args = parser.parse_args()
    results, years = read_results(args.files)
    if args.tags == "most_freq":
        tags = [u'MD', u'VAN', u'VBP', u'Q', u'C', u'VB', u'NS', u'VBD', u'FW', u'PRO$',
                u'NPR', u'ADV', u'.', u'ADJ', u',', u'CONJ', u'D', u'PRO', u'P', u'N']       
    else:
        tags = intersection(*[r[0] for r in results])
    metrics = {"f1": f1, "precision": precision, "recall": recall}
    to_plotly_json(results, years, metrics[args.metric], tags, args.output)
    

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
