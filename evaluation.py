
import codecs

from collections import Counter

from sklearn.metrics import confusion_matrix
from utils import deserialize_results, plot_conf_matrix
from penn_data import pos_from_range

y_true, y_pred, labels = deserialize_results("models/1500.json")
#cm = confusion_matrix(y_true, y_pred)
#plot_conf_matrix(cm, labels= labels)

def counts_per_tag(sents):
    return Counter(tag for sent in sents for word, tag in sent)


counts = dict()
test = (1400, 1500, 2000)
test_counts = counts_per_tag(pos_from_range(*test))
for start in range(1400, 1850, 50):
    train = (start, start + 100, 10000)
    sents = pos_from_range(*train)
    train_counts = counts_per_tag(sents)
    counts[start] = train_counts

    # todo: measure correlation between per tag accuracy and counts over time
