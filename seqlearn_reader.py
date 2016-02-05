
from itertools import tee
from collections import Counter

from sklearn.feature_extraction import FeatureHasher


def features(tokens, i, tags=None, window=2):
    "feature function"
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


def sent_sequences(sents, feature_fn, labels, lengths, tag_features=None):
    "transforms a iterator over sents into a generator over features per token"
    for s in sents:
        length = len(s)
        words, tags = zip(*s)
        labels.extend(tags)
        lengths.append(length)
        for i in range(length):
            yield features(words, i, tags if tag_features else None)


def load_data_hasher(sents, features=features, tag_features=None, compute_num_feats=False):
    "computes seqlearn input data using a feature hasher"
    y, lengths = [], []
    X_raw = sent_sequences(sents, features, y, lengths, tag_features)
    if compute_num_feats:
        X_raw, X_raw2 = tee(X_raw)
        counter = Counter(feats for feats in X_raw2)
        n_features = int(len(counter) * 1.25)
        fh = FeatureHasher(input_type='string', n_features=n_features)
    else:
        fh = FeatureHasher(input_type='string')
    X = fh.transform(X_raw)
    return X, y, lengths


def load_data_dict(sents, dv, features=features, tag_features=None):
    "does not overwrite feature vectorizer"
    y, lengths = [], []
    X_raw = sent_sequences(sents, features, y, lengths, tag_features)
    if hasattr(dv, 'vocabulary_'):
        X = dv.transform(Counter(w) for w in X_raw)
    else:
        X = dv.fit_transform(Counter(w) for w in X_raw)
    return X, y, lengths
