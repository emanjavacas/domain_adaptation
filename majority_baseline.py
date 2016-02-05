from __future__ import print_function

from collections import Counter

from utils import serialize_results, split_sents
from penn_data import pos_from_range

from sklearn.metrics import accuracy_score


class MajorityBaseline(object):
    def __init__(self):
        self.word2tag = None
        self.unknown = 'UNK'

    def predict(self, X):
        if not self.word2tag:
            raise ValueError("Model hasn't been trained yet")
        return [self.word2tag.get(w.lower(), self.unknown) for w in X]

    def fit(self, X, y, **kwargs):
        counts = Counter(zip(X, y))
        results = {}
        for (w, t), c in counts.items():
            if w in results:
                current_max_count = counts.get((w, t), 0)
                results[w] = t if c > current_max_count else results[w]
            else:
                results[w] = t
        self.word2tag = results
        self.classes_ = results.values()


if __name__ == "__main__":
    test = (1400, 1500, 4000)
    X_test, y_test = split_sents(pos_from_range(*test))
    for start in range(1400, 1850, 50):
        print("training on range", start)
        train = (start, start + 100, 30000)
        X_train, y_train = split_sents(pos_from_range(*train))
        clf = MajorityBaseline()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        prefix = "models/" + str(start) + "_baseline"
        labels = clf.classes_
        serialize_results(prefix, y_true=y_test, y_pred=y_pred, labels=labels)
        print("Training set size", len(X_train))        
        print("Test set size", len(X_test))       
        print("OOV tokens", Counter(y_pred)['UNK'] / float(len(y_pred)))
        print("Accuracy: %.3f" % (100 * accuracy_score(y_test, y_pred)))


    
    
