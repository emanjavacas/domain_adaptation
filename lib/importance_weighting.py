import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression as log
from sklearn.naive_bayes import BernoulliNB as nb

a = {1: (["rec.sport.hockey", "comp.sys.ibm.pc.hardware"],
         ["rec.sport.baseball", "comp.sys.mac.hardware"]),
     2: (["rec.sport.hockey", "sci.crypt"],
         ["rec.sport.baseball", "sci.electronics"]),
     3: (["talk.politics.guns", "sci.electronics"],
         ["talk.politics.mideast", "sci.med"]),
     4: (["comp.graphics", "talk.politics.misc"],
         ["comp.windows.x", "talk.religion.misc"])}

for pair_idx in a:
    S = fetch_20newsgroups(subset="train", shuffle=True,
                           categories=a[pair_idx][0])
    T = fetch_20newsgroups(subset="train", shuffle=True,
                           categories=a[pair_idx][1])
    test = fetch_20newsgroups(subset="test", shuffle=True,
                              categories=a[pair_idx][1])
    # transform data
    y_train, y_test = S.target, test.target
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(S.data)
    X_test = vectorizer.transform(test.data)
    # domain classifier
    X_dom = vectorizer.transform(S.data + T.data)
    y_dom = [0 for _ in range(len(S.data))] + [1 for _ in range(len(T.data))]
    y_dom = np.array(y_dom)

    weights = np.zeros(y_train.shape)
    clf = log()

    clf.fit(X_dom[::2], y_dom[::2])  # predict odds based on even
    for i in range(1, X_train.shape[0], 2):
        weights[i] = clf.predict_proba(X_dom[i])[0][1]
    clf.fit(X_dom[1::2], y_dom[1::2])  # predict evens based on odd
    for i in range(0, X_train.shape[0], 2):
        weights[i] = clf.predict_proba(X_dom[i])[0][1]

    clf = nb()
    clf.fit(X_train, y_train)
    y_bl = clf.predict(X_test)
    print "bl: ", clf.score(X_test, y_test)

    clf.fit(X_train, y_train, sample_weight=weights)
    y_sys = clf.predict(X_test)
    print "sys: ", clf.score(X_test, y_test)
