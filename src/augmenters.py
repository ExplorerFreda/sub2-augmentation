import collections
import copy
import random
import regex
from nltk import Tree
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

from data import Sentence


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
            subseq_words = self.dataset[sent_id].words[
                start_id:start_id+length]
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


class DependencyParsingAugmenter(Augmenter):
    def __init__(self, dataset, n_gram=4):
        super(DependencyParsingAugmenter, self).__init__(dataset)
        self.k = n_gram
        self.build_subtree_table(dataset)

    def reset(self):
        self.dataset = copy.deepcopy(self.original_dataset)

    def build_subtree_table(self, dataset):
        # augment with length restriction
        # TODO (freda) consider removing the restriction? 
        self.subtrees = collections.defaultdict(list)
        self.tree_infos = list()
        self.subtree_list = list()
        for i, sent in enumerate(dataset):
            # drop if sentence has X-edges
            x_edge = False
            for k, j in enumerate(sent.ids):
                assert k == j - 1
                parent = sent.deps[k]
                if parent == 0:
                    continue
                left, right = min(j, parent), max(j, parent)
                for id_p in range(left + 1, right):
                    parent_p = sent.deps[id_p - 1]
                    if parent_p < left or parent_p > right:
                        x_edge = True
                if x_edge:
                    break
            if x_edge:
                continue
            # collect info 
            tree_info = collections.defaultdict(dict)
            for j in sent.ids:
                tree_info[j]['left'] = j
                tree_info[j]['right'] = j
            tree_info[0] = {'left': 1e10, 'right': 0}
            for k, j in enumerate(sent.ids):
                parent = sent.deps[k]
                tree_info[parent]['left'] = min(
                    tree_info[j]['left'], tree_info[parent]['left']
                )
            for k, j in reversed(list(enumerate(sent.ids))):
                parent = sent.deps[k]
                tree_info[parent]['right'] = max(
                    tree_info[j]['right'], tree_info[parent]['right']
                )
            for k, j in enumerate(sent.ids):
                length = tree_info[j]['right'] - tree_info[j]['left'] + 1
                if length <= 1:
                    continue
                label = sent.dep_labels[k]
                self.subtrees[length, label].append(
                    [i, k, tree_info[j]['left']-1, tree_info[j]['right']]
                )
            self.tree_infos.append(tree_info)
        for key in self.subtrees:
            for subtree in self.subtrees[key]:
                self.subtree_list.append(subtree + list(key))

    def augment(self, size=None):
        # self.ids, self.words, self.tags, self.deps, self.dep_labels
        if size is None:
            size = len(self.dataset) * 2
        bar = tqdm(range(size - len(self.dataset)))
        bar.set_description(f'Running {type(self)}')
        while len(self.dataset) < size:
            position = random.randint(0, len(self.subtree_list) - 1)
            original_subtree = self.subtree_list[position]
            length = original_subtree[-2]
            label = original_subtree[-1]
            substitution_position = random.randint(
                0, len(self.subtrees[length, label]) - 1
            )
            substitution_subtree = self.subtrees[
                length, label][substitution_position]
            if substitution_subtree == original_subtree[:4]:
                continue
            ids = list(self.dataset[original_subtree[0]].ids)
            words = list(self.dataset[original_subtree[0]].words)
            tags = list(self.dataset[original_subtree[0]].tags)
            deps = list(self.dataset[original_subtree[0]].deps)
            dep_labels = list(self.dataset[original_subtree[0]].dep_labels)
            orig_range = original_subtree[2:4]
            sub_range = substitution_subtree[2:4]
            words[orig_range[0]:orig_range[1]] = copy.deepcopy(
                self.dataset[substitution_subtree[0]].words[
                    sub_range[0]:sub_range[1]
                ]
            )
            tags[orig_range[0]:orig_range[1]] = copy.deepcopy(
                self.dataset[substitution_subtree[0]].tags[
                    sub_range[0]:sub_range[1]
                ]
            )
            orig_parent = self.dataset[original_subtree[0]].deps[
                original_subtree[1]
            ]
            sub_deps = [
                x - sub_range[0] for x in self.dataset[
                    substitution_subtree[0]].deps[sub_range[0]:sub_range[1]]
            ]
            deps[orig_range[0]:orig_range[1]] = [
                x + orig_range[0] for x in sub_deps
            ]
            deps[substitution_subtree[1] - sub_range[0] + orig_range[0]] = \
                orig_parent
            dep_labels[orig_range[0]:orig_range[1]] = copy.deepcopy(
                self.dataset[substitution_subtree[0]].dep_labels[
                    sub_range[0]:sub_range[1]
                ]
            )
            new_sent = Sentence.from_info(ids, words, tags, deps, dep_labels)
            self.dataset.data.append(new_sent)
        return self.dataset


class POSTagBaselineAugmenter(Augmenter):
    def __init__(self, dataset, 
            language_model='bert-large-cased-whole-word-masking'):
        super(POSTagBaselineAugmenter, self).__init__(dataset)
        self.tokenizer = AutoTokenizer.from_pretrained(language_model)
        self.model = AutoModelForMaskedLM.from_pretrained(language_model)

    def reset(self):
        self.dataset = copy.deepcopy(self.original_dataset)

    def augment(self, size=None):
        if size is None:
            size = len(self.dataset) * 2
        bar = tqdm(range(size - len(self.dataset)))
        bar.set_description(f'Running {type(self)}')
        while len(self.dataset) < size:
            sent_id = random.randint(0, len(self.dataset) - 1)
            sub_sentence = copy.deepcopy(self.dataset[sent_id])
            words = list(sub_sentence.words)
            position = random.randint(0, len(words)-1)
            orig_token = words[position]
            # only perform SUB on word tokens
            if not regex.match('.*[a-z].*', orig_token):
                continue
            # compute substitution and output a different sub candidate
            words[position] = self.tokenizer.mask_token
            inputs = self.tokenizer(' '.join(words), return_tensors='pt')
            subtoken_position = (
                inputs['input_ids'][0] == self.tokenizer.mask_token_id
            ).long().argmax().item()
            labels = inputs['input_ids']
            outputs = self.model(**inputs, labels=labels)[1][0]
            while True:
                subtoken_id = outputs[subtoken_position].argmax().item()
                subtoken = self.tokenizer.convert_ids_to_tokens(subtoken_id)
                if subtoken != orig_token:
                    break
                else:
                    outputs[subtoken_position][subtoken_id] = -1e10
            words[position] = subtoken
            sub_sentence.words = tuple(words)
            self.dataset.data.append(sub_sentence)
            bar.update()
        return self.dataset


class DependencyParsingBaselineAugmenter(POSTagBaselineAugmenter):
    def __init__(self, *args, **kwargs):
        super(DependencyParsingBaselineAugmenter, self).__init__(
            *args, **kwargs
        )


class POSTagJointAugmenter(Augmenter):
    def __init__(self, dataset):
        super(POSTagJointAugmenter, self).__init__(dataset)
        self.sub_augmenter = POSTagAugmenter(dataset)
        self.synonym_augmenter = POSTagBaselineAugmenter(dataset)

    def reset(self):
        self.dataset = copy.deepcopy(self.original_dataset)

    def augment(self, size=None):
        if size is None:
            size = len(self.dataset) * 2
        size_per_method = len(self.dataset) + (size - len(self.dataset)) // 2
        sub_augs = self.sub_augmenter.augment(size_per_method)
        synonym_augs = self.synonym_augmenter.augment(size_per_method)
        self.dataset += sub_augs[len(self.dataset):] + \
            synonym_augs[len(self.dataset):]
        return self.dataset


class DependencyParsingJointAugmenter(POSTagJointAugmenter):
    def __init__(self, dataset):
        super(POSTagJointAugmenter, self).__init__(dataset)
        self.sub_augmenter = DependencyParsingAugmenter(dataset)
        self.synonym_augmenter = DependencyParsingBaselineAugmenter(dataset)

    def reset(self):
        self.dataset = copy.deepcopy(self.original_dataset)

    def augment(self, size=None):
        if size is None:
            size = len(self.dataset) * 2
        size_per_method = len(self.dataset) + (size - len(self.dataset)) // 2
        sub_augs = self.sub_augmenter.augment(size_per_method)
        synonym_augs = self.synonym_augmenter.augment(size - size_per_method)
        self.dataset += sub_augs[len(self.dataset):] + \
            synonym_augs[len(self.dataset):]
        return self.dataset


if __name__ == "__main__":
    # Joint augmenter unit test
    from data import UniversalDependenciesDataset
    dataset = UniversalDependenciesDataset(
        '../data/*/*/en*dev*conllu',
        '../data/universal-dependencies-1.2/tags.txt'
    )
    augmenter = DependencyParsingJointAugmenter(dataset)
    augmented_dataset = augmenter.augment()
    from IPython import embed; embed(using=False)

    # POS tagging augmenter unit test
    from data import UniversalDependenciesDataset
    dataset = UniversalDependenciesDataset(
        '../data/*/*/en*dev*conllu',
        '../data/universal-dependencies-1.2/tags.txt'
    )
    augmenter = POSTagBaselineAugmenter(dataset)
    augmented_dataset = augmenter.augment()
    from IPython import embed; embed(using=False)

    # Dep parsing unit test
    from data import UniversalDependenciesDataset
    dep_dataset = UniversalDependenciesDataset(
        '../data/universal_treebanks_v2.0/std/en/en-universal-dev.conll',
        '../data/universal_treebanks_v2.0/std/tags.txt'
    )
    augmenter = DependencyParsingAugmenter(dep_dataset)
    augmented_dataset = augmenter.augment()
    from IPython import embed; embed(using=False)

    # PTB augmenter unit test
    from data import PTBDataset
    random.seed(115)
    sst_dataset = PTBDataset(
        f'../data/sst/train.txt', use_spans=True, span_min_length=4
    )
    sst_augmenter = SSTAugmenter(sst_dataset)
    sst_augmented_dataset = sst_augmenter.augment()
    from IPython import embed; embed(using=False)
   
    # POS tagging augmenter unit test
    from data import UniversalDependenciesDataset
    dataset = UniversalDependenciesDataset(
        '../data/*/*/en*train*conllu',
        '../data/universal-dependencies-1.2/tags.txt'
    )
    augmenter = POSTagAugmenter(dataset)
    augmented_dataset = augmenter.augment()
    from IPython import embed; embed(using=False)