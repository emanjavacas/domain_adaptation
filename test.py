from node import Node, from_list

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
          ['A3312', 'tag']],
         ['A332', 'tag']],
        ['A33',                 # repeat for testing
         ['A331', 'tag']],
        ['A35',
         ['A351', 'tag'],
         ['A352',
          ['A3521', 'tag'],
          ['A3522', 'tag']]]],
       ['A4', 'tag']]

root = Node(tag=ex2[0])
from_list(ex2[1:], root=root)
