import argparse
import benepar
from nltk import Tree
from tqdm import tqdm


def add_label(tree, label):
    if isinstance(tree, str):
        return tree
    children = [add_label(child, label) for child in tree]
    tree = Tree(label + '-' + tree.label(), children)
    return tree


parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', type=str)
parser.add_argument('--output', '-o', type=str)
parser.add_argument('--model', '-m', type=str, default='benepar_en2_large')
parser.add_argument('--style', '-s', type=str, help='tree|json', default='tree')
args = parser.parse_args()

c_parser = benepar.Parser(args.model)
if args.style == 'tree':
    with open(args.output, 'w') as fout:
        for line in tqdm(open(args.input).readlines()):
            senti_tree = Tree.fromstring(line)
            sent = ' '.join(senti_tree.leaves())
            cons_tree = c_parser.parse(sent)
            # add full sentiment label to each internal node
            cons_tree = add_label(cons_tree, senti_tree.label())
            cons_tree_str = ' '.join(str(cons_tree).replace('\n', ' ').split())
            print(cons_tree_str, file=fout)
elif args.style == 'json':
    pass
else:
    raise Exception(f'Data style {args.style} not supported.')
