
from utils import pickle_this
from seqlearn_reader import load_data_hasher, load_data_dict
from penn_data import pos_from_range

from seqlearn.perceptron import StructuredPerceptron
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction import DictVectorizer


def model_for_range_hasher(train=(1700, 1800, 10000), test=(1400, 1500, 2000)):
    train_data = pos_from_range(*train)
    test_data = pos_from_range(*test)
    X_train, y_train, lengths_train = load_data_hasher(train_data)
    X_test, y_test, lengths_test = load_data_hasher(test_data)
    clf = StructuredPerceptron(verbose=True, max_iter=10)
    clf.fit(X_train, y_train, lengths_train)
    return clf, None, X_test, y_test, lengths_test


def model_for_range_dict(train=(1700, 1800, 10000), test=(1400, 1500, 2000)):
    train_data = pos_from_range(*train)
    test_data = pos_from_range(*test)
    dv = DictVectorizer()
    X_train, y_train, lengths_train = load_data_dict(train_data, dv)
    X_test, y_test, lengths_test = load_data_dict(test_data, dv)
    clf = StructuredPerceptron(verbose=True, max_iter=10)
    clf.fit(X_train, y_train, lengths_train)
    return clf, dv, X_test, y_test, lengths_test


if __name__ == '__main__':
    test = (1400, 1500, 2000)
    for start in range(1400, 1850, 50):
        train = (start, start + 100, 10000)
        clf, featurizer, X_test, y_test, lengths_test = \
            model_for_range_hasher(train, test)
        y_pred = clf.predict(X_test, lengths_test)
        prefix = "models/" + str(start)
        labels = clf.classes_
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        pickle_this(prefix + '_y_test', y_test)
        pickle_this(prefix + '_y_pred', y_pred)
        pickle_this(prefix + '_labels', labels)
#        pickle_this(prefix + '_clf', clf)
        print("Accuracy: %.3f" % (100 * accuracy_score(y_test, y_pred)))
