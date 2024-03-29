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
    
    def reset(self):
        self.dataset = copy.deepcopy(self.original_dataset)


class POSTagAugmenter(Augmenter):
    def __init__(self, dataset, n_gram=4):
        super(POSTagAugmenter, self).__init__(dataset)
        self.k = n_gram
        self.build_ngram_table(dataset, n_gram)

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
        

class CParseAugmenter(Augmenter):
    def __init__(
                self, dataset, span_min_length=4, span_max_length=20
            ):
        super(CParseAugmenter, self).__init__(dataset)
        self.rg = (span_min_length, span_max_length)
        self.build_subtree_table(dataset, self.rg)

    @staticmethod
    def collect_subtrees(tree, left=0):
        if isinstance(tree, str):
            return
        current_left = left
        for child in tree:
            for item in CParseAugmenter.collect_subtrees(child, current_left):
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
            try:
                new_tree = self.substitute_tree(
                    self.dataset.trees[idx], 0, left, length, sub_subtree
                )
            except:
                print('erorr performing substitution.')
                from IPython import embed; embed(using=False)
            self.dataset.trees.append(new_tree)
            bar.update()
        self.dataset.add_spans(self.dataset.trees[original_tree_size:])
        return self.dataset

    @staticmethod
    def substitute_tree(tree, left, goal_left, length, subtree):
        tree_label = None if isinstance(tree, str) else tree.label()
        len_tree = 1 if isinstance(tree, str) else len(tree.leaves())
        if left == goal_left and len_tree == length and \
                tree_label == subtree.label():
            return copy.deepcopy(subtree)
        elif isinstance(tree, str):
            return tree
        elif left > goal_left:
            return copy.deepcopy(tree)
        current_left = left
        new_children = list()
        for child in tree:
            new_child = CParseAugmenter.substitute_tree(
                child, current_left, goal_left, length, subtree
            )
            new_children.append(new_child)
            len_child = 1 if isinstance(child, str) else len(child.leaves())
            current_left += len_child
        return Tree(tree.label(), new_children)

    @staticmethod
    def rebuild(tree, words):
        if isinstance(tree, str):
            assert len(words) == 1
            return words[0]
        assert len(tree.leaves()) == len(words)
        left = 0
        new_children = list()
        for child in tree:
            n_leaves = len(child.leaves()) if isinstance(child, Tree) else 1
            new_child = CParseAugmenter.rebuild(
                child, words[left:left+n_leaves]
            )
            left += n_leaves
            new_children.append(new_child)
        return Tree(tree.label(), new_children)


class CParseLengthFreeAugmenter(CParseAugmenter):
    def __init__(self, *args, **kwargs):
        super(CParseLengthFreeAugmenter, self).__init__(*args, **kwargs)

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
                self.subtree_table[subtree.label()].append(subtree_info)
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
                0, len(self.subtree_table[subtree.label()]) - 1
            )
            sub_idx, sub_left, sub_length, sub_subtree = self.subtree_table[
                subtree.label()][sub_position]
            assert subtree.label() == sub_subtree.label()
            if ' '.join(sub_subtree.leaves()) == ' '.join(subtree.leaves()):
                continue
            try:
                new_tree = self.substitute_tree(
                    self.dataset.trees[idx], 0, left, length, sub_subtree
                )
            except:
                print('erorr performing substitution.')
                from IPython import embed; embed(using=False)
            self.dataset.trees.append(new_tree)
            bar.update()
        self.dataset.add_spans(self.dataset.trees[original_tree_size:])
        return self.dataset


class CParseSynonymAugmenter(CParseAugmenter):
    def __init__(
                self, dataset, 
                language_model='bert-large-cased-whole-word-masking'
            ):
        super(CParseSynonymAugmenter, self).__init__(dataset)
        self.tokenizer = AutoTokenizer.from_pretrained(language_model)
        self.model = AutoModelForMaskedLM.from_pretrained(language_model).cpu()

    def augment(self, size=None):
        if size is None:
            size = len(self.dataset) * 2
        original_tree_size = len(self.dataset.trees)
        bar = tqdm(range(size - len(self.dataset)))
        bar.set_description(f'Running {type(self)}')
        while len(self.dataset.trees) < size:
            tree_id = random.randint(0, len(self.dataset) - 1)
            sub_tree = copy.deepcopy(self.dataset.trees[tree_id])
            words = list(sub_tree.leaves())
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
            new_tree = self.rebuild(sub_tree, words)
            self.dataset.trees.append(new_tree)
            bar.update()
        self.dataset.add_spans(self.dataset.trees[original_tree_size:])
        return self.dataset


class CParseRandomAugmenter(CParseAugmenter):
    def __init__(self, dataset):
        super(CParseRandomAugmenter, self).__init__(dataset)

    def augment(self, size=None):
        if size is None:
            size = len(self.dataset) * 2
        original_tree_size = len(self.dataset.trees)
        bar = tqdm(range(size - len(self.dataset)))
        bar.set_description(f'Running {type(self)}')
        while len(self.dataset.trees) < size:
            tree_id = random.randint(0, len(self.dataset) - 1)
            sub_tree = copy.deepcopy(self.dataset.trees[tree_id])
            words = list(sub_tree.leaves())
            random.shuffle(words)
            new_tree = self.rebuild(sub_tree, words)
            self.dataset.trees.append(new_tree)
            bar.update()
        self.dataset.add_spans(self.dataset.trees[original_tree_size:])
        return self.dataset


class CParseSpanAugmenter(CParseAugmenter):
    def __init__(self, dataset):
        super(CParseSpanAugmenter, self).__init__(dataset)
        self.example_table = self.build_example_table()

    def build_example_table(self):
        example_table = collections.defaultdict(list)
        for tree in self.dataset.trees:
            label = tree.label()
            example_table[label].append(tree)
        return example_table

    def augment(self, size=None):
        if size is None:
            size = len(self.dataset) * 2
        original_tree_size = len(self.dataset.trees)
        bar = tqdm(range(size - len(self.dataset)))
        bar.set_description(f'Running {type(self)}')
        while len(self.dataset.trees) < size:
            tree_id = random.randint(0, len(self.dataset) - 1)
            sub_tree = copy.deepcopy(self.dataset.trees[tree_id])
            label = sub_tree.label()
            words = list(sub_tree.leaves())
            sub_left = random.randint(0, len(words) - 1)
            sub_right = random.randint(sub_left + 1, len(words))
            candidate_id = random.randint(0, len(self.example_table[label]) - 1)
            candidate_tree = copy.deepcopy(self.example_table[label][candidate_id])
            candidate_words = candidate_tree.leaves()
            can_left = random.randint(0, len(candidate_words) - 1)
            can_right = random.randint(can_left + 1, len(candidate_words))
            new_words = words[:sub_left] + candidate_words[can_left:can_right] + words[sub_right:]
            new_tree = Tree(label, new_words)
            self.dataset.trees.append(new_tree)
            bar.update()
        self.dataset.add_spans(self.dataset.trees[original_tree_size:])
        return self.dataset


class CParseLengthSpanAugmenter(CParseAugmenter):
    def __init__(self, dataset):
        super(CParseLengthSpanAugmenter, self).__init__(dataset)
        self.example_table = self.build_example_table()

    def build_example_table(self):
        example_table = collections.defaultdict(list)
        for tree in self.dataset.trees:
            label = tree.label()
            example_table[label].append(tree)
        return example_table

    def augment(self, size=None):
        if size is None:
            size = len(self.dataset) * 2
        original_tree_size = len(self.dataset.trees)
        bar = tqdm(range(size - len(self.dataset)))
        bar.set_description(f'Running {type(self)}')
        while len(self.dataset.trees) < size:
            tree_id = random.randint(0, len(self.dataset) - 1)
            sub_tree = copy.deepcopy(self.dataset.trees[tree_id])
            label = sub_tree.label()
            words = list(sub_tree.leaves())
            sub_left = random.randint(0, len(words) - 1)
            sub_right = random.randint(sub_left + 1, len(words))
            candidate_id = random.randint(0, len(self.example_table[label]) - 1)
            candidate_tree = copy.deepcopy(self.example_table[label][candidate_id])
            candidate_words = candidate_tree.leaves()
            can_left = random.randint(0, len(candidate_words) - 1)
            can_right = can_left + sub_right - sub_left 
            if can_right > len(candidate_words):
                continue
            new_words = words[:sub_left] + candidate_words[can_left:can_right] + words[sub_right:]
            new_tree = Tree(label, new_words)
            self.dataset.trees.append(new_tree)
            bar.update()
        self.dataset.add_spans(self.dataset.trees[original_tree_size:])
        return self.dataset


class DependencyParsingAugmenter(Augmenter):
    def __init__(self, dataset, n_gram=4):
        super(DependencyParsingAugmenter, self).__init__(dataset)
        self.k = n_gram
        self.build_subtree_table(dataset)

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

    def augment(self, size=None):
        if size is None:
            size = len(self.dataset) * 2
        size_per_method = len(self.dataset) + (size - len(self.dataset)) // 2
        sub_augs = self.sub_augmenter.augment(size_per_method)
        synonym_augs = self.synonym_augmenter.augment(size_per_method)
        self.dataset.data += sub_augs.data[len(self.dataset):] + \
            synonym_augs.data[len(self.dataset):]
        return self.dataset


class DependencyParsingJointAugmenter(POSTagJointAugmenter):
    def __init__(self, dataset):
        super(DependencyParsingJointAugmenter, self).__init__(dataset)
        self.sub_augmenter = DependencyParsingAugmenter(dataset)
        self.synonym_augmenter = DependencyParsingBaselineAugmenter(dataset)

    def augment(self, size=None):
        if size is None:
            size = len(self.dataset) * 2
        size_per_method = len(self.dataset) + (size - len(self.dataset)) // 2
        sub_augs = self.sub_augmenter.augment(size_per_method)
        synonym_augs = self.synonym_augmenter.augment(size - size_per_method)
        self.dataset.data += sub_augs.data[len(self.dataset):] + \
            synonym_augs.data[len(self.dataset):]
        return self.dataset


class POSTagRandomAugmenter(Augmenter):
    def __init__(self, *args, **kwargs):
        super(POSTagRandomAugmenter, self).__init__(*args, **kwargs)

    def augment(self, size=None):
        if size is None:
            size = len(self.dataset) * 2
        bar = tqdm(range(size - len(self.dataset)))
        bar.set_description(f'Running {type(self)}')
        while len(self.dataset) < size:
            sent_id = random.randint(0, len(self.dataset) - 1)
            sub_sentence = copy.deepcopy(self.dataset[sent_id])
            new2orig = [i for i in range(len(sub_sentence.words))]
            random.shuffle(new2orig)
            orig2new = {i: k for k, i in enumerate(new2orig)}
            orig2new[-1] = -1  # root
            pos = [sub_sentence.tags[i] for _, i in enumerate(new2orig)]
            words = [sub_sentence.words[i] for _, i in enumerate(new2orig)]
            deps = [
                orig2new[sub_sentence.deps[i]-1]+1 
                for _, i in enumerate(new2orig)
            ]
            dep_labels = [
                sub_sentence.dep_labels[i] for _, i in enumerate(new2orig)
            ]
            sub_sentence.words = words
            sub_sentence.tags = pos
            sub_sentence.deps = deps
            sub_sentence.dep_labels = dep_labels
            self.dataset.data.append(sub_sentence)
            bar.update()
        return self.dataset


class DependencyParsingRandomAugmenter(POSTagRandomAugmenter):
    def __init__(self, *args, **kwargs):
        super(DependencyParsingRandomAugmenter, self).__init__(*args, **kwargs)


if __name__ == "__main__":
    # augmenter unit test

    from data import PTBDataset

    random.seed(115)
    sst_dataset = PTBDataset(
        f'../data/sst/train_c.txt', use_spans=False
    )
    sst_rand_augmenter = CParseRandomAugmenter(sst_dataset)
    sst_augmented_dataset = sst_rand_augmenter.augment(100000)
    from IPython import embed; embed(using=False)

    random.seed(115)
    sst_dataset = PTBDataset(
        f'../data/sst/train_c.txt', use_spans=False
    )
    sst_syno_augmenter = CParseSynonymAugmenter(sst_dataset)
    sst_augmented_dataset = sst_syno_augmenter.augment(100000)
    from IPython import embed; embed(using=False)

    random.seed(115)
    sst_dataset = PTBDataset(
        f'../data/sst/train_cl.txt', use_spans=True, span_min_length=4
    )
    sst_free_augmenter = CParseLengthFreeAugmenter(sst_dataset)
    sst_augmented_dataset = sst_free_augmenter.augment(100000)
    from IPython import embed; embed(using=False)

    random.seed(115)
    sst_dataset = PTBDataset(
        f'../data/sst/train_cl.txt', use_spans=True, span_min_length=4
    )
    sst_augmenter = CParseAugmenter(sst_dataset)
    sst_augmented_dataset = sst_augmenter.augment(100000)
    from IPython import embed; embed(using=False)

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
   
    # POS tagging augmenter unit test
    from data import UniversalDependenciesDataset
    dataset = UniversalDependenciesDataset(
        '../data/*/*/en*train*conllu',
        '../data/universal-dependencies-1.2/tags.txt'
    )
    augmenter = POSTagAugmenter(dataset)
    augmented_dataset = augmenter.augment()
    from IPython import embed; embed(using=False)

    from data import UniversalDependenciesDataset
    ud_dataset = UniversalDependenciesDataset(
        '../data/ud-treebanks-v2.6/UD_English-*/en_ewt-ud-train.conllu', 
        '../data/ud-treebanks-v2.6/tags.txt'
    )
    ud_rand_augmenter = POSTagRandomAugmenter(ud_dataset)
    ud_rand_augmenter.augment()
    from IPython import embed; embed(using=False)
