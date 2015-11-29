import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB
from sklearn.base import clone


class SelfTraining():
    """
    It makes the following assumptions:
    X_train = L_train + 2 * U_train
    y_train = y_train + 2 * [0]
    """
    def __init__(self, learner=Perceptron(), iters=5, pool=100,
                 select=lambda y_hat: True if float(y_hat) == 1 else False):
        self.learner = learner
        self.iters = iters
        self.pool = pool
        self.select = select

    def fit(self, X_train, y_train, U_size):
        L_size = X_train.shape[0] - U_size * 2
        weights = [1] * L_size + [0] * U_size * 2

        def M():
            """
            Maximisation based on labeled data
            Weights out unlabeled data.
            """
            self.learner.fit(X_train, y_train, sample_weights=weights)

        def E():
            picked = np.random.randint(L_size, L_size + U_size, self.pool)
            for i in picked:
                y_hat = self.learner.predict(X_train[i, :])
                if self.select(y_hat) == 1:
                    weights[i] = 1
                else:
                    weights[i + U_size] = 1
        M()
        for _ in range(self.iters):
            E()
            M()

    def predict(self, X):
        return self.learner.predict(X)

    def score(self, X, y):
        return self.learner.score(X, y)


class CoTraining():
    def __init__(self, learner=Perceptron(), iters=5, pool=100,
                 select=lambda x: True if len(set(x)) == 1 else False):
        self.learner = learner
        self.iters = iters
        self.pool = pool
        self.select = select
        self.learners = []

    def fit(self, X_train, X2_train, y_train, U_size):
        # todo: clever analysis of conditional independence
        if len(set([i.shape[0] for i in X2_train] + X_train)) != 1:
            raise ValueError("Unequal sample size across views")
        L_size = X2_train[0].shape[0] - 2 * U_size
        weights = [1] * L_size + [0] * 2 * U_size
        n_learners = len(X2_train)

        def M():
            self.learner.fit(X_train, y_train, sample_weights=weights)
            for l in range(n_learners):
                self.learners[l] = clone(self.learner)
                self.learners[l].fit(X2_train[l], y_train,
                                     sample_weights=weights)

        def E():
            picked = np.random.randint(L_size, L_size + U_size, self.pool)
            for i in picked:
                y_hats = []
                for l in range(n_learners):
                    y_hat = self.learners[l].predict(X2_train[l][i, :])
                    y_hats.append(y_hat)
                    if self.select(y_hats):
                        weights[i] = 1
                    else:
                        weights[i + U_size] = 1

        M()
        for _ in range(self.iters):
            E()
            M()

    def predict(self, X):
        return self.learner.predict(X)

    def score(self, X, y):
        return self.learner.score(X, y)


class TriTraining():
    def __init__(self, learner=Perceptron(), iters=5, pool=100):
        self.n_learners = 3
        self.learner = learner
        self.learners = [clone(learner) for _ in range(self.n_learners)]
        self.iters = iters
        self.pool = pool

    def fit(self, X_train, y_train, U_size):
        L_size = X_train.shape[0] - 2 * U_size
        weights = [[1] * L_size + [0] * 2 * U_size for _ in self.n_learners]

        def M():
            for i, learner in enumerate(self.learners):
                learner.fit(X_train, y_train, sample_weights=weights[i])

        def E():
            for l in range(self.n_learners):
                for u in range(L_size, L_size + U_size):
                    j, k = list(set(range(self.n_learners)) - set([l]))
                    yj = self.learners[j].predict(X_train[u])
                    yk = self.learners[k].predict(X_train[u])
                    if yj == yk:
                        weights[l][u] = 1
                    else:
                        weights[l][u + U_size] = 1

        M()
        for _ in range(self.iters):
            E()
            M()

    def predict(self, X):
        return self.learners[0].predict(X)

    def score(self, X, y):
        return [learner.score(X, y) for learner in self.learners]


class EM():
    """Dilable soft self-training"""
    def __init__(self, learner=BernoulliNB(), iters=5):
        self.learner = learner
        self.iters = iters

    def fit(self, X_train, y_train, U_size):
        L_size = X_train.shape[0] - U_size * 2
        weights = [1] * L_size + [0] * U_size * 2

        def M():
            self.learner.fit(X_train, y_train, sample_weights=weights)

        def E():
            picked = np.random.randint(L_size, L_size + U_size, U_size)
            for i in picked:
                y_hat = self.learner.predict(X_train[i, :])
                if float(y_hat) == 1:
                    proba = self.learner.predict_proba(X_train[i, :]).flatten()
                    weights[i] = int(max(proba))
                    weights[i + U_size] = int(min(proba))
                else:
                    weights[i] = int(min(proba))
                    weights[i + U_size] = int(max(proba))

        M()
        for _ in range(self.iters):
            E()
            M()

    def predict(self, X):
        return self.learner.predict(X)

    def score(self, X, y):
        return self.learner.score(X, y)
