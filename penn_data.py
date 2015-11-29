import os
import codecs

root = "/Users/quique/corpora/PENN-CORPORA/"
main_dirs = ["PPCEME-RELEASE-2/corpus/",
             "PPCMBE-RELEASE-1/corpus/",
             "PPCME2-RELEASE-3/corpus/"]
info_file = root + "corpus_data.csv"


def read_info(in_fn=info_file):
    result = {}
    with open(in_fn, "r") as f:
        next(f)
        for l in f:
            l = l.strip()
            fname, date, genre, wc, genre2, span = l.split(",")
            result[fname.strip('"')] = tuple([date, genre, wc, genre2, span])
    return result


def pos_from_file(fname):
    with codecs.open(fname, "r", "utf-8") as f:
        sent = []
        for l in f:
            l = l.strip()
            if l.startswith("<") or l.startswith("{"):  # ignore meta
                continue
            if not l and not sent:  # blank lines
                continue
            elif not l:
                yield sent
                sent = []
                continue
            word, token = l.split("/")
            sent.append((word, token))


def tree_from_file(fname):
    with codecs.open(fname, "r", "utf-8") as f:
        tree_string = ""
        for l in f:
            l = l.strip()
            if not l:
                yield parse(tree_string)
                tree_string = ""
            else:
                tree_string += l


def get_pos_sents():
    target = "pos"
    for d in main_dirs:
        for f in os.listdir(root + d + target):
            for pos in pos_from_file(root + d + target + "/" + f):
                yield pos


# kindly taken from http://norvig.com/lispy.html
def tokenize(tree_string):
    return tree_string.replace('(', ' ( ').replace(')', ' ) ').split()


def read_from_tokens(tokens):
    if len(tokens) == 0:
        raise SyntaxError("unexpected EOF")
    token = tokens.pop(0)
    if token == '(':
        L = []
        while tokens[0] != ')':
            L.append(read_from_tokens(tokens))
        tokens.pop(0)
        return L
    elif token == ')':
        raise SyntaxError("unexpected )")
    else:
        return atom(token)


def atom(token):
    return token


def parse(string):
    return read_from_tokens(tokenize(string))


def get_psd_sents():
    target = "psd"
    for d in main_dirs:
        for f in os.listdir(root + d + target):
            for tree in tree_from_file(root + d + target + "/" + f):
                yield tree
