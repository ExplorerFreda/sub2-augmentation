import collections
import copy
import random
from nltk import Tree
from tqdm import tqdm


class Augmenter(object):
    def __init__(self, dataset):
        self.original_dataset = dataset
        self.dataset = copy.deepcopy(dataset)


class POSTagAugmenter(Augmenter):
    def __init__(self, dataset, n_gram=4):
        super(POSTagAugmenter, self).__init__(dataset)
        self.k = n_gram
        self.build_ngram_table(dataset, n_gram)

    def reset(self):
        self.dataset = copy.deepcopy(self.original_dataset)

    def build_ngram_table(self, dataset, k):
        self.ngram_table = collections.defaultdict(list)
        for i, sent in enumerate(dataset):
            assert len(sent.words) == len(sent.tags)
            for j in range(k):
                for s in range(len(sent.tags) - j):
                    subseq_tag = '_'.join(sent.tags[s:s+j+1])
                    self.ngram_table[subseq_tag].append((i, s, j+1))
        self.ngram_positions = list()
        for key in self.ngram_table:
            self.ngram_positions.extend(self.ngram_table[key])

    def augment(self, size=None):
        if size is None:
            size = len(self.dataset) * 2
        bar = tqdm(range(size - len(self.dataset)))
        bar.set_description(f'Running {type(self)}')
        while len(self.dataset) < size:
            position = random.randint(0, len(self.ngram_positions) - 1)
            sent_id, start_id, length = self.ngram_positions[position]
            subseq_tag = '_'.join(
                self.dataset[sent_id].tags[start_id:start_id+length]
            )
            sub_position = random.randint(
                0, len(self.ngram_table[subseq_tag]) - 1
            )
            sub_sent_id, sub_start_id, sub_length = \
                self.ngram_table[subseq_tag][sub_position]
            assert sub_length == length
            subseq_words = self.dataset[sent_id].words[start_id:start_id+length]
            sub_subseq_words = self.dataset[sub_sent_id].words[
                sub_start_id:sub_start_id+sub_length]
            if subseq_words == sub_subseq_words:
                continue
            sub_sentence = copy.deepcopy(self.dataset[sent_id])
            sub_sentence.words = sub_sentence.words[:start_id] + \
                sub_subseq_words + \
                sub_sentence.words[start_id+length:]
            self.dataset.data.append(sub_sentence)
            bar.update()
        return self.dataset
        

class SSTAugmenter(Augmenter):
    def __init__(self, dataset, span_min_length=4, span_max_length=20):
        super(SSTAugmenter, self).__init__(dataset)
        self.rg = (span_min_length, span_max_length)
        self.build_subtree_table(dataset, self.rg)

    def reset(self):
        self.dataset = copy.deepcopy(self.original_dataset)

    @staticmethod
    def collect_subtrees(tree, left=0):
        if isinstance(tree, str):
            return
        current_left = left
        for child in tree:
            for item in SSTAugmenter.collect_subtrees(child, current_left):
                yield item
            child_len = 1 if isinstance(child, str) else len(child.leaves())
            current_left += child_len
        yield tree, left, len(tree.leaves())

    def build_subtree_table(self, dataset, span_length_range):
        self.subtree_table = collections.defaultdict(list)
        self.all_subtrees = list()
        span_min_length, span_max_length = span_length_range
        for i, tree in enumerate(tqdm(dataset.trees)):
            tree_info = self.collect_subtrees(tree)
            for subtree, left, length in tree_info:
                if length < span_min_length or length > span_max_length:
                    continue
                subtree_info = (i, left, length, copy.deepcopy(subtree))
                self.subtree_table[subtree.label(), length].append(
                    subtree_info
                )
                self.all_subtrees.append(subtree_info)

    def augment(self, size=None):
        if size is None:
            size = len(self.dataset.trees) * 2
        original_tree_size = len(self.dataset.trees)
        bar = tqdm(range(size - len(self.dataset.trees)))
        bar.set_description(f'Running {type(self)}')
        while len(self.dataset.trees) < size:
            position = random.randint(0, len(self.all_subtrees) - 1)
            idx, left, length, subtree = self.all_subtrees[position]
            sub_position = random.randint(
                0, len(self.subtree_table[subtree.label(), length]) - 1
            )
            sub_idx, sub_left, sub_length, sub_subtree = self.subtree_table[
                subtree.label(), length][sub_position]
            assert sub_length == length
            if ' '.join(sub_subtree.leaves()) == ' '.join(subtree.leaves()):
                continue
            new_tree = self.substitute_tree(
                self.dataset.trees[idx], 0, left, length, sub_subtree
            )
            self.dataset.trees.append(new_tree)
            bar.update()
        self.dataset.add_spans(self.dataset.trees[original_tree_size:])
        return self.dataset

    @staticmethod
    def substitute_tree(tree, left, goal_left, length, subtree):
        tree_label = None if isinstance(tree, str) else tree.label()
        len_tree = 1 if isinstance(tree, str) else len(tree.leaves())
        if left == goal_left and len_tree == len(subtree.leaves()):
            assert tree.label() == subtree.label()
            return copy.deepcopy(subtree)
        elif isinstance(tree, str):
            return tree
        elif left >= goal_left + length:
            return copy.deepcopy(tree)
        current_left = left
        new_children = list()
        for child in tree:
            new_child = SSTAugmenter.substitute_tree(
                child, current_left, goal_left, length, subtree
            )
            new_children.append(new_child)
            len_child = 1 if isinstance(child, str) else len(child.leaves())
            current_left += len_child
        return Tree(tree.label(), new_children)

    
if __name__ == "__main__":
    from data import PTBDataset
    random.seed(115)
    sst_dataset = PTBDataset(
        f'../data/sst/train.txt', use_spans=True, span_min_length=4
    )
    sst_augmenter = SSTAugmenter(sst_dataset)
    sst_augmented_dataset = sst_augmenter.augment()
    from IPython import embed; embed(using=False)
   
    from data import UniversalDependenciesDataset
    dataset = UniversalDependenciesDataset(
        '../data/*/*/en*train*conllu',
        '../data/universal-dependencies-1.2/tags.txt'
    )
    augmenter = POSTagAugmenter(dataset)
    augmented_dataset = augmenter.augment()
    from IPython import embed; embed(using=False)