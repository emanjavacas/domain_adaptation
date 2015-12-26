
import random


def take(g, n):
    c = 0
    while c < n:
        yield g.next()
        c += 1


def shuffle(seq, seed=448):
    return sorted(seq, key=lambda k: random.random())
