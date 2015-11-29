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
            if l.startswith("<") or l.startswith("{"):
                continue
            if not l and not sent:
                continue
            if not l:
                yield sent
                sent = []
                continue
            word, token = l.split("/")
            sent.append((word, token))


def get_pos_sents():
    target = "pos"
    for d in main_dirs:
        for f in os.listdir(root + d + target):
            # text_id = f.split("/")[-1].split(".")[-2]
            for pos in pos_from_file(root + d + target + "/" + f):
                yield pos


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


def get_psd_sents():
    target = "psd"
    for d in main_dirs:
        for f in os.listdir(root + d + target):
            # text_id = f.split("/")[-1].split(".")[-2]
            for tree in tree_from_file(root + d + target + "/" + f):
                yield tree


class Node(object):
    def __init__(self, tag, parent=None, children=None):
        self.tag = tag
        self.parent = parent
        self.siblings = []
        self.children = [] if not children else children
        self.is_terminal = True if isinstance(self.children, list) else False

    def get_parent(self, tag):
        curr = self
        while curr.parent and curr.parent.tag is not tag:
            curr = curr.parent
        return curr

    def get_child(self, tag):
        return [node for node in self.iter_children() if node.tag is tag]

    def add_child(self, node):
        if not isinstance(node, Node):
            raise ValueError("Cannot append node of type: [%s]" % type(node))
        if not self.is_terminal:
            raise ValueError("Cannot append node to terminal")
        else:
            for child in self.children:
                child.siblings.append(node)
            node.siblings.extend(self.children)
            self.children.append(node)

    def iter_children(self):
        if self.is_terminal:
            yield self.children
        else:
            for child in self.children:
                yield child
                if not child.is_terminal:
                    for child in child.iter_children():
                        yield child

    def __repr__(self):
        if not self.is_terminal:
            s = "<Node: %s; parent: %s>" % (self.tag, self.parent)
        else:
            s = "<Node: %s, %s; parent: %s>" \
                % (self.tag, self.children, self.parent)
        return s

ex = [u'IP-MAT',
      [u'NP-SBJ',
       [u'D', u'Y=e='],
       [u'ADJS', u'best'],
       [u'PP', [u'P', u'of'], [u'NP', [u'D', u'y=e='], [u'NS', u'men']]]],
      [u'BED', u'was'],
      [u'NP-OB1',
       [u'NP', [u'NPR', u'Lord'], [u'NPR', u'Antrim']],
       [u',', u','],
       [u'CONJP', [u'NP', [u'NPR', u'Lord'], [u'NPR', u'Anglese']]],
       [u',', u','],
       [u'CONJP',
        [u'CONJ', u'and'],
        [u'NP', [u'NPR', u'Lord'], [u'NPR', u'Essex']]]],
      [u'.', u'.']]

ex2 = ['A',
       ['A1',
        ['A11', 'tag'],
        ['A12', 'tag'],
        ['A13',
         ['A131', 'tag'],
         ['A132',
          ['A1321', 'tag'],
          ['A1322', 'tag']]]],
       ['A2', 'tag'],
       ['A3',
        ['A31',
         ['A311', 'tag'],
         ['A312', 'tag']],
        ['A32', 'tag'],
        ['A33',
         ['A331',
          ['A3311', 'tag'],
          ['A3312', 'tag']]],
        ['A33',
         ['A331', 'tag']],
        ['A35',
         ['A351', 'tag'],
         ['A352',
          ['A3521', 'tag'],
          ['A3522', 'tag']]]],
       ['A4', 'tag']]

ex3 = ['IP-MAT',
 ['NP-LFD',
  ['D', '+Tys'],
  ['N', 'vertu'],
  [',', ','],
  ['IP-MAT-PRN',
   ['NP-SBJ', ['D', '+tat']],
   ['BEP', 'is'],
   ['IP-INF',
    ['TO', 'to'],
    ['VB', 'seye'],
    ['PP',
     ['P', 'of'],
     ['NP', ['N', 'maydenhood'], ['CONJ', 'or'], ['N', 'chastite']]]]]],
 [',', ','],
 ['NP-SBJ-RSP', ['PRO', 'hit']],
 ['BEP', 'is'],
 ['NP-OB1',
  ['NP', ['D', 'a'], ['ADJ', 'wylful'], ['N', 'sacryfyse']],
  ['CONJP',
   ['CONJ', 'and'],
   ['NP',
    ['D', 'an'],
    ['N', 'offryngge'],
    ['PP', ['P', 'to'], ['NP', ['NPR', 'God']]],
    ['ADJP', ['ADJ', 'vre'], ['CONJ', 'and'], ['ADJ', 'liberal']]]],
  [',', ','],
  ['CP-REL',
   ['WPP-1', ['P', 'to'], ['WNP', ['D', '+te'], ['WPRO', 'whyche']]],
   ['C', '0'],
   ['IP-SUB',
    ['IP-SUB',
     ['PP', '*T*-1'],
     ['NP-SBJ', ['Q', 'no'], ['N', 'lawe']],
     ['VBP', 'dryf+t']],
    [',', ','],
    ['CONJP',
     ['IP-SUB',
      ['PP', '*T*-1'],
      ['NP-SBJ', ['Q', 'no'], ['N', 'nyede']],
      ['VBP', 'constreyne+t']]],
    [',', ','],
    ['CONJP',
     ['IP-SUB',
      ['PP', '*T*-1'],
      ['NP-SBJ', ['Q', 'non'], ['N', 'heste']],
      ['VBP', 'bynt']]]]]],
 ['.', ';']]


def traverse(tree):
    if isinstance(tree, list):
        for it in tree:
            parent = tree[0]
            for subit in traverse(it):
                # if subit[-1] != parent or subit[-1] == ',':
                yield [parent] + subit
    else:
        yield [tree]


def getelements(lst, key, res=None):
    if res is None:
        res = []
    for e in lst:
        if isinstance(e, list):
            if e[0] == key:
                res.append(e)
            else:
                getelements(e, key, res)
    return res


def from_list(lst, root=None, depth=0):
    print depth
    lst = list(lst)
    if root is None:
        root = Node(tag=lst.pop(0))
    for e in lst:
        if is_terminal(e):
            tag, children = e
            print "terminal", tag, "with root", root.tag
            depth -= 1
            root.add_child(Node(tag=tag, children=children, parent=root))
        else:
            e = list(e)
            tag, children = e.pop(0), e
            print "non terminal", tag, "with root", root.tag
            depth += 1
            root = Node(tag=tag, parent=root)
            from_list(children, root=root, depth=depth)
    return root


def is_terminal(e):
    if isinstance(e, list) and \
        len(e) == 2 and all([type(i) in (unicode, str) for i in e]):
        return True
    return False


def traverse(tree):
    if is_terminal(tree):
        yield [tuple(tree)]
    elif isinstance(tree, list):
        for it in tree:
            parent = tree[0]
            for subit in traverse(it):
                if subit and isinstance(subit[-1], tuple):
                    yield [parent] + subit
    else:
        yield []


def branches(root, branch):
    if not branch:
        return
    if not root.children or branch[0] != [i.tag for i in root.children][-1]:
        if isinstance(branch[0], tuple):
            node = Node(tag=branch[0][0], children=branch[0][1], parent=root)
        else:
            node = Node(tag=branch[0], parent=root)
        root.add_child(node)
    else:
        node = [n for n in root.children if n.tag == branch[0]][-1]
    branches(node, branch[1:])

d = Node(tag=u'IP-MAT')
for b in traverse(ex2):
    branches(d, b[1:])
