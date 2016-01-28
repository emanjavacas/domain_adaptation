
import argparse

from utils import serialize_results
from seqlearn_reader import load_data_hasher, load_data_dict
from penn_data import pos_from_range

from seqlearn.perceptron import StructuredPerceptron
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction import DictVectorizer


def model_for_range_hasher(train=(1700, 1800, 10000), test=(1400, 1500, 2000)):
    train_data = pos_from_range(*train)
    test_data = pos_from_range(*test)
    X_train, y_train, lengths_train = load_data_hasher(train_data, feature_tags=None)
    X_test, y_test, lengths_test = load_data_hasher(test_data, feature_tags=None)
    clf = StructuredPerceptron(verbose=True, max_iter=10)
    clf.fit(X_train, y_train, lengths_train)
    return clf, None, X_test, y_test, lengths_test


def model_for_range_dict(train=(1700, 1800, 10000), test=(1400, 1500, 2000)):
    train_data = pos_from_range(*train)
    test_data = pos_from_range(*test)
    dv = DictVectorizer()
    X_train, y_train, lengths_train = load_data_dict(train_data, dv, feature_tags=None)
    X_test, y_test, lengths_test = load_data_dict(test_data, dv, feature_tags=None)
    clf = StructuredPerceptron(verbose=True, max_iter=10)
    clf.fit(X_train, y_train, lengths_train)
    return clf, dv, X_test, y_test, lengths_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run training")
    parser.add_argument("prefix")
    args = parser.parse_args()
    test = (1400, 1450, 2000)
    for start in range(1450, 1850, 50):
        train = (start, start + 100, 20000)
        clf, featurizer, X_test, y_test, lengths_test = \
            model_for_range_hasher(train, test)
        y_pred = clf.predict(X_test, lengths_test)
        labels = clf.classes_
        #cm = confusion_matrix(y_test, y_pred, labels=labels)
        serialize_results(args.prefix, y_true=y_test, y_pred=y_pred, labels=labels)
        print("Training set size", len(X_train))        
        print("Test set size", len(X_test))       
        print("Accuracy: %.3f" % (100 * accuracy_score(y_test, y_pred)))


