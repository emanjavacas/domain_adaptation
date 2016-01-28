
import random
<<<<<<< HEAD
import os
import re
=======
>>>>>>> a08a6cfb1855c739dacc3ae74f0ea6f36bb16d74
import codecs
import cPickle as p
import json
from datetime import datetime
from collections import defaultdict


def take(g, n):
    c = 0
    while c < n:
        yield g.next()
        c += 1


def destruct(d, *args):
    return (d[arg] for arg in args)
<<<<<<< HEAD


def intersection(*seqs):
    a, rest = list(seqs).pop(), seqs
    return [set(a) & set(s) for s in rest][0]


def split_sents(sents):
    X = []
    y = []
    for sent in sents:
        words, tags = zip(*sent)
        X.extend(words)
        y.extend(tags)
    return X, y


=======


def intersection(*seqs):
    a, rest = list(seqs).pop(), seqs
    return [set(a) & set(s) for s in rest][0]


def split_sents(sents):
    X = []
    y = []
    for sent in sents:
        words, tags = zip(*sent)
        X.extend(words)
        y.extend(tags)
    return X, y


>>>>>>> a08a6cfb1855c739dacc3ae74f0ea6f36bb16d74
def shuffle_seq(seq, seed=448):
    return sorted(seq, key=lambda k: random.random())


def pickle_this(fname, obj):
    fname += '_' + str(datetime.time(datetime.now())) + '.pickle'
    with open(fname, 'w') as f:
        p.dump(obj, f)

        
def unpickle_this(fname):
    with open(fname, 'r') as f:
        return p.load(f)

    
def serialize_results(fname, y_true, y_pred, labels):
    result = {"y_true": list(y_true),
              "y_pred": list(y_pred),
              "labels": list(labels)}
    fname += "_" + str(datetime.time(datetime.now())) + ".json"
    with codecs.open(fname, "w+", "utf-8") as f:
        json.dump(result, f)

        
def deserialize_results(fname):
<<<<<<< HEAD
    try:
        with codecs.open(fname, "r+", "utf-8") as f:
            results = json.load(f)
            if type(results) == dict:
                return results["y_true"], results["y_pred"], results["labels"]
            elif type(results) == list:
                return results[0], results,[1], results[2]
    except Exception as e:
        raise ValueError("Reading exception with file: " + fname)

def read_year(fname):
    fname = os.path.basename(fname)
    return re.findall(r'\d+', fname)[0]


def read_results(files, read_year_fn=read_year):
    results = [(deserialize_results(f), read_year_fn(f)) for f in sorted(files)]
    return zip(*results)
=======
    with codecs.open(fname, "r+", "utf-8") as f:
        results = json.load(f)
        return results["y_true"], results["y_pred"], results["labels"]
>>>>>>> a08a6cfb1855c739dacc3ae74f0ea6f36bb16d74


def tags_intersection(results):
    return intersection(*(result[0] for result in results))
