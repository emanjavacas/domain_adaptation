
from seqlearn_reader import features, load_data
from penn_data import pos_from_range

from seqlearn.perceptron import StructuredPerceptron
from sklearn.metrics import accuracy_score


def model_for_range(train=(1700, 1800, 8000), test=(1400, 1500, 2000)):
    train_data = pos_from_range(*train)
    test_data = pos_from_range(*test)
    X_train, y_train, lengths_train = load_data(train_data, features)
    X_test, y_test, lengths_test = load_data(test_data, features)
    clf = StructuredPerceptron(verbose=True, max_iter=10)
    clf.fit(X_train, y_train, lengths_train)
    return clf, X_test, y_test, lengths_test


if __name__ == '__main__':
    clf, X_test, y_test, lengths_test = model_for_range()
    y_pred = clf.predict(X_test, lengths_test)
    print("Accuracy: %.3f" % (100 * accuracy_score(y_test, y_pred)))
