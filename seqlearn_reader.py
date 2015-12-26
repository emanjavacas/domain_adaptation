
from sklearn.feature_extraction import FeatureHasher


def features(words, tags, i, window=2):
    length = len(words)
    target = words[i]
    yield 'bias'
    if target[0].isupper():
        yield 'CAP'
    yield 'prefix:{}' + target[0]
    first, last = max(0, i - window), min(length - 1, i + window + 1)
    for idx in range(first, last):
        word = words[idx]
        yield 'word[%d]:{}%s' % (idx - i, word.lower())
        yield 'suffix[%d]:{}%s' % (idx - i, word[-3:])
        if idx < i:
            yield 'tag[%d]:{}%s' % (idx - i, tags[idx])


def sent_sequences(sents, feature_fn, labels, lengths):
    for s in sents:
        length = len(s)
        words, tags = zip(*s)
        labels.extend(tags)
        lengths.append(length)
        for i in range(length):
            yield features(words, tags, i)


def load_data(sents, feature_fn):
    y, lengths = [], []
    X_raw = sent_sequences(sents, feature_fn, y, lengths)
    fh = FeatureHasher(input_type='string')
    X = fh.transform(X_raw)
    return X, y, lengths
