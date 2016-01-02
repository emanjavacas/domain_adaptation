
from itertools import tee
from collections import Counter

from sklearn.feature_extraction import FeatureHasher


def features(tokens, i, tags, window=2):
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


def clf_predict(tokens, dv, clf, window=2):
    "incrementally predicts computing tag-based features"
    tags = []
    for i in range(len(tokens)):
        feats = features(tokens, i, tags, window)
        tag = clf.predict(dv.transform(Counter(feats))).item()
        tags.extend([tag])
    assert len(tags) == len(tokens)
    return tags


def sent_sequences(sents, feature_fn, labels, lengths):
    "transforms a iterator over sents into a generator over features per token"
    for s in sents:
        length = len(s)
        words, tags = zip(*s)
        labels.extend(tags)
        lengths.append(length)
        for i in range(length):
            yield features(words, i, tags)


def load_data_hasher(sents, features=features, compute_feats=False):
    "computes seqlearn input data using a feature hasher"
    y, lengths = [], []
    X_raw = sent_sequences(sents, features, y, lengths)
    if compute_feats:
        X_raw, X_raw2 = tee(X_raw)
        counter = Counter(feats for feats in X_raw2)
        n_features = int(len(counter) * 1.25)
        fh = FeatureHasher(input_type='string', n_features=n_features)
    else:
        fh = FeatureHasher(input_type='string')
    X = fh.transform(X_raw)
    return X, y, lengths


def load_data_dict(sents, dv, features=features):
    "does not overwrite feature vectorizer"
    y, lengths = [], []
    X_raw = sent_sequences(sents, features, y, lengths)
    if hasattr(dv, 'vocabulary_'):
        X = dv.transform(Counter(w) for w in X_raw)
    else:
        X = dv.fit_transform(Counter(w) for w in X_raw)
    return X, y, lengths

# # train1
# sents = pos_from_range(1400, 1500, 5000)
# dv1 = DictVectorizer()
# X, y, lengths = load_data_dict(sents, dv1)
# clf1 = StructuredPerceptron(verbose=True, max_iter=10)
# clf1.fit(X, y, lengths)

# # train2
# sents = pos_from_range(1400, 1500, 5000)
# dv2 = DictVectorizer()
# X, y, lengths = load_data_dict(sents, dv2)
# clf2 = StructuredPerceptron(verbose=True, max_iter=10)
# clf2.fit(X, y, lengths)

# # test
# sents = pos_from_range(1500, 1600, 1000)
# X2, y2, lengths2 = [], [], []
# for sent in sents:
#     words, tags = zip(*sent)
#     lengths = len(words)
#     lengths2.append(lengths)
#     X2.extend([features(words, i, []) for i in range(len(words))])
#     y2.extend(tags)

# X2 = dv2.transform(Counter(w) for w in X2)
# y_pred = clf2.predict(X2, lengths2)

# sents = pos_from_range(1500, 1600, 1000)
# sents = shuffle_seq(sents)[:100]

# real_tags = []
# with_tags = []
# without_tags = []
# for sent in sents:
#     words, tags = zip(*sent)
#     real_tags.extend(tags)

#     pred = clf_predict(words, dv, clf)
#     with_tags.extend(pred)

#     feats = (features(words, i, []) for i in range(len(words)))
#     pred = clf.predict(dv.transform(Counter(f) for f in feats))
#     without_tags.extend(pred)
