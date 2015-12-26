

class Node(object):
    def __init__(self, tag, parent=None, children=None):
        self.tag = tag
        self.parent = parent
        self.siblings = []
        self.children = [] if not children else children
        self.is_terminal = False if isinstance(self.children, list) else True

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
        if self.is_terminal:
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
        if self.is_terminal:
            s = "Node<type='%s'; value='%s'>" % (self.tag, self.children)
        else:
            s = "Node<type='%s'; children=%d>" % (self.tag, len(self.children))
        return s


def is_terminal(e):
    "auxiliary function for list-trees"
    if isinstance(e, (tuple, list)) and \
       len(e) == 2 and all([isinstance(i, (unicode, str)) for i in e]):
        return True
    return False


def from_list(lst, root=None):
    """
    recursively populates a root node with children from a tree
    structure defined in a nested list
    """
    lst = list(lst)
    if not lst:
        return
    for e in lst:
        if is_terminal(e):
            tag, children = e
            print "terminal", tag, "with root", root.tag
            root.add_child(Node(tag=tag, children=children, parent=root))
        else:
            e = list(e)
            tag, children = e.pop(0), e
            print "non terminal", tag, "with root", root.tag
            newroot = Node(tag=tag, parent=root)
            root.add_child(newroot)
            from_list(children, root=newroot)


def leaf_branches(tree):
    """
    return a generator of paths to terminals (branches to leaves)
    for a given root node (non-terminal node)
    """
    if is_terminal(tree):
        yield [tuple(tree)]
    elif isinstance(tree, list):
        for it in tree:
            parent = tree[0]
            for subit in leaf_branches(it):
                if subit and isinstance(subit[-1], tuple):
                    yield [parent] + subit
    else:
        yield []


def tree_from_list(lst):
    "utility function"
    tree = lst[0]
    root_tag = tree[0]
    assert isinstance(root_tag, (str, unicode))
    root = Node(tag=root_tag)
    from_list(tree[1:], root=root)
    return root
