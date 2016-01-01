
from collections import Counter

from sklearn.feature_extraction import FeatureHasher, DictVectorizer


def features(tokens, tags, i, window=2):
    length = len(tokens)
    target = tokens[i]
    yield 'bias'
    if target[0].isupper():
        yield 'CAP'
    yield 'prefix:{%s}' % target[0]
    first, last = max(0, i - window), min(length - 1, i + window + 1)
    for idx in range(first, last):
        word = tokens[idx]
        yield 'word[%d]:{%s}' % (idx - i, word.lower())
        yield 'suffix[%d]:{%s}' % (idx - i, word[-3:])
        if idx < i and tags:
            yield 'tag[%d]:{%s}' % (idx - i, tags[idx])


def clf_predict(tokens, dv, clf, window=2):
    tags = []
    for i in range(len(tokens)):
        feats = features(tokens, tags, i, window)


def sent_sequences(sents, feature_fn, labels, lengths):
    for s in sents:
        length = len(s)
        words, tags = zip(*s)
        labels.extend(tags)
        lengths.append(length)
        for i in range(length):
            yield features(words, tags, i)


def load_data(sents, features=features):
    y, lengths = [], []
    X_raw = sent_sequences(sents, features, y, lengths)
    fh = FeatureHasher(input_type='string')
    X = fh.transform(X_raw)
    return X, y, lengths


def load_data_dict(sents, dv, features=features):
    y, lengths = [], []
    X_raw = sent_sequences(sents, features, y, lengths)
    X = dv.fit_transform(Counter(s) for s in X_raw)
    return X, y, lengths


# from penn_data import pos_from_range
# from seqlearn.perceptron import StructuredPerceptron

sents = pos_from_range(1400, 1500, 50)
dv = DictVectorizer()
X, y, lengths = load_data_dict(sents, dv)
split_lengths = int(len(lengths) * 0.9)
split = sum(lengths[:split_lengths])
X_train, y_train, lengths_train = X[:split,], y[:split], lengths[:split_lengths]
X_test, y_test, lengths_test = X[split:,], y[split:], lengths[split_lengths:]
clf = StructuredPerceptron(verbose=True, max_iter=10)
clf.fit(X_train, y_train, lengths_train)

feats = 

# dv = DictVectorizer()
# D = [{'dog': 1, 'cat': 2, 'elephant': 4}, {'dog': 2, 'run': 5}]
# C = [{'bear': 1, 'butterfly': 2, 'deer': 4}, {'deer': 2, 'apple': 5}]
# DD = dv.fit_transform(D)
# CC = dv.transform(C)
