
from utils import take, shuffle_seq

try:
    from config import root
except:
    raise Warning("set global variable ``root'' to the appropriate value")

import os
import re
import glob
import codecs

genre_mapping = {
    '"BIBLE"': '"RELIGION"',
    '"BIOGRAPHY_AUTO"': '"DIARY"',
    '"BIOGRAPHY_LIFE_OF_SAINT"': '"NARRATIVE"',
    '"BIOGRAPHY_OTHER"': '"NARRATIVE"',
    '"DIARY_PRIV"': '"DIARY"',
    '"DRAMA_COMEDY"': '"FICTION"',
    '"EDUC_TREATISE"': '"NONFICTION"',
    '"FICTION"': '"FICTION"',
    '"HANDBOOK_ASTRO"': '"NONFICTION"',
    '"HANDBOOK_MEDICINE"': '"NONFICTION"',
    '"HANDBOOK_OTHER"': '"NONFICTION"',
    '"HISTORY"': '"NARRATIVE"',
    '"HOMILY"': '"RELIGION"',
    '"HOMILY_POETRY"': '"RELIGION"',
    '"LAW"': '"NONFICTION"',
    '"LETTERS_NON-PRIV"': '"LETTERS"',
    '"LETTERS_PRIV"': '"LETTERS"',
    '"PHILOSOPHY"': '"NARRATIVE"',
    '"PHILOSOPHY/FICTION"': '"NARRATIVE"',
    '"PROCEEDINGS_TRIAL"': '"NONFICTION"',
    '"RELIG_TREATISE"': '"RELIGION"',
    '"ROMANCE"': '"FICTION"',
    '"RULE"': '"NONFICTION"',
    '"SCIENCE_MEDICINE"': '"NONFICTION"',
    '"SCIENCE_OTHER"': '"NONFICTION"',
    '"SERMON"': '"RELIGION"',
    '"TRAVELOGUE"': '"DIARY"'
}

INF = float('inf')


def read_info(in_fn=os.path.join(root, "corpus_data.csv")):
    result = {}
    with open(in_fn, "r") as f:
        next(f)
        for l in f:
            l = l.strip()
            fname, date, genre, wc, genre2, span = l.split(",")
            result[fname.strip('"')] = tuple([date, genre, wc, genre2, span])
    return result


def abs_path(basename, ext):
    fnames = glob.glob(root + "*RELEASE*/corpus/*/*." + ext)
    fname = ".".join([basename, ext])
    for f in fnames:
        if fname in f:
            return f

def grep_file(prefix, ext):
    fnames = glob.glob(root + "*RELEASE*/corpus/*/*." + ext)
    for f in fnames:
        basename = os.path.basename(f)
        if basename.startswith(prefix):
            return f


def files_in_range(from_y, to_y, ext='pos'):
    info = read_info()
    result = []
    for f, row in info.items():
        year = int(row[0].split('-')[0])
        if from_y <= year < to_y:            
            result.append(grep_file(f, ext))
    return result


def simplify_tag(tag):
    if '+' in tag:
        tag = tag.split('+')[0]
    tag = re.sub(r'[0-9]+$', '', tag)
    return tag


def pos_from_file(fname, rem_id=True, simple_tags=True):
    with codecs.open(fname, "r", "utf-8") as f:
        sent = []
        for l in f:
            l = l.strip()
            if l.startswith("<") or l.startswith("{"):  # ignore metadata
                continue
            if not l and not sent:  # blank lines
                continue
            elif not l:
                yield sent
                sent = []
                continue
            word, tag = l.split("/")
            if tag == 'ID' and rem_id:
                continue
            tag = simplify_tag(tag) if simple_tags else tag
            sent.append((word, tag))


def pos_from_files(files, max_sents=INF, rem_id=True, shuffle=False, shuffle_seed=448):
    sents = (sent for f in files for sent in pos_from_file(f, rem_id=rem_id))
    return take(iter(shuffle_seq(sents, shuffle_seed)) if shuffle else sents, max_sents)


def pos_from_range(from_y, to_y, max_sents=INF, rem_id=True, shuffle=False, shuffle_seed=448):
    files = files_in_range(from_y, to_y)
    sents = pos_from_files(files, max_sents=max_sents, rem_id=rem_id, shuffle=shuffle)
    return sents


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


def get_pos(simple_tags=True):
    fnames = glob.glob(root + "*RELEASE*/corpus/*/*.pos")
    for f in fnames:
        for pos in pos_from_file(f, simple_tags=simple_tags):
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


def get_psd():
    fnames = glob.glob(root + "*RELEASE*/corpus/*/*.psd")
    for f in fnames:
        for tree in tree_from_file(f):
            yield tree
