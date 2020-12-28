import json
import nltk
import regex
from glob import glob
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer 


class Sentence(object):
    def __init__(self, conll_data=None):
        self.conll_data = conll_data
        if conll_data is not None:
            self.ids, self.words, self.tags, self.deps, self.dep_labels = \
                self.extract_info(conll_data)
    
    @classmethod
    def from_info(cls, ids, words, tags, deps, dep_labels):
        sent = cls()
        sent.ids, sent.words, sent.tags, sent.deps, sent.dep_labels = \
            ids, words, tags, deps, dep_labels
        return sent
    
    def __str__(self):
        return json.dumps(
            {
                'ids': self.ids,
                'words': self.words, 
                'tags': self.tags, 
                'deps': self.deps, 
                'dep_labels': self.dep_labels
            }
        )

    @staticmethod
    def extract_info(conll_data):
        filtered_conll_data = map(
            lambda y: [int(y[0]), y[1], y[3], int(y[6]), y[7]],
            filter(lambda x: regex.match(r'[0-9]+$', x[0]), conll_data)
        )
        ids, words, tags, deps, dep_labels = zip(*filtered_conll_data)
        return ids, words, tags, deps, dep_labels

    def print(self, fout):
        for i, ids in enumerate(self.ids):
            info = ['_'] * 10
            info[0] = str(self.ids[i])
            info[1] = self.words[i]
            info[3] = self.tags[i]
            info[6] = str(self.deps[i])
            info[7] = self.dep_labels[i]
            print('\t'.join(info), file=fout)
        print(file=fout)


class UniversalDependenciesDataset(Dataset):
    def __init__(self, data_path_template, tagset_path):
        super(UniversalDependenciesDataset, self).__init__()
        self.tag2id = {k.strip(): i for i, k in enumerate(open(tagset_path))}
        self.data = list()
        current_conll_data = list()
        for path in glob(data_path_template):
            for line in open(path):
                if self.is_data_line(line):
                    current_conll_data.append(line.strip().split('\t'))
                elif len(current_conll_data) > 0:
                    self.data.append(Sentence(current_conll_data))
                    current_conll_data = list()
            if len(current_conll_data) > 0:
                self.data.append(Sentence(current_conll_data))
    
    def print(self, file_name):
        with open(file_name, 'w') as fout:
            for sent in self.data:
                sent.print(fout)
            fout.close()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def cut(self, length=None):
        if length is None:
            return 
        else:
            self.data = self.data[:length]

    @staticmethod
    def is_data_line(line):
        items = line.split('\t') 
        return len(items) > 1 and regex.match(r'[0-9]+', items[0]) is not None

    @classmethod
    def collate_fn(cls, batch):
        return batch

    @classmethod
    def __pad__(cls, data, padding_token):
        max_length = max(len(x) for x in data)
        padded_data = list(data)
        for i in range(len(padded_data)):
            padded_data[i] = list(padded_data[i]) + [padding_token] * (
                max_length - len(padded_data[i])
            )
        return padded_data
    
    def get_tag_ids(self, sentences):
        tag_ids = [
            [self.tag2id.get(tag, self.tag2id['X']) for tag in sent.tags]
            for sent in sentences
        ]
        return self.__pad__(tag_ids, -1)

    def filter_length_for_model(self, model='xlm-roberta-large'):
        tokenizer = AutoTokenizer.from_pretrained(model)
        data = list()
        for i, sent in enumerate(tqdm(self.data)):
            length = len(tokenizer.tokenize(' '.join(sent.words)))
            if length < tokenizer.model_max_length:
                data.append(sent)
        self.data = data


# Constituency Parsing Dataset, including SST
class PTBDataset(Dataset):
    def __init__(
                self, data_path_template, use_spans=True, span_min_length=1,
                preproc_method='g', collapse_unary=True
            ):
        super(PTBDataset, self).__init__()
        self.trees = list()
        self.collapse_unary = collapse_unary
        for path in glob(data_path_template):
            for line in open(path):
                tree = nltk.Tree.fromstring(line)
                if 'g' in preproc_method:  # keep only class labels for roots
                    tree.set_label(tree.label().split('-')[0])
                if 'p' in preproc_method:  # keep only phrase labels for NTs
                    for s in tree.subtrees():
                        s.set_label(s.label().split('-')[-1])
                if collapse_unary:
                    tree.collapse_unary()
                self.trees.append(tree)
        # preproc spans
        self.use_spans = use_spans
        self.span_min_length = span_min_length
        self.spans = list()
        self.add_spans(self.trees)
    
    def add_spans(self, trees):
        if self.use_spans is True:
            for tree in trees:
                spans = list(self.extract_spans(tree))
                for span in spans:
                    left = span[0]
                    right = span[1]
                    if right - left >= self.span_min_length or \
                            right - left == len(tree.leaves()):
                        self.spans.append((' '.join(tree.leaves()), *span))
        else:
            for tree in trees:
                self.spans.append(
                    (
                        ' '.join(tree.leaves()), 
                        0, len(tree.leaves()),
                        tree.label()
                    )
                )
    
    @staticmethod
    def extract_spans(tree, left=0):
        if isinstance(tree, nltk.Tree): 
            current_left = left
            for child in tree:
                for span in PTBDataset.extract_spans(child, current_left):
                    yield span
                current_left += 1 if isinstance(
                    child, str) else len(child.leaves())
            yield (left, left + len(tree.leaves()), tree.label())

    def __len__(self):
        return len(self.spans)

    def __getitem__(self, index):
        return self.spans[index]
    
    @classmethod
    def collate_fn(cls, batch):
        return list(zip(*batch))
    

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    # PTB Test
    phrases = set()
    for split in ['train', 'dev', 'test']:
        sst_dataset = PTBDataset(
            f'../data/sst/{split}.txt', 
            use_spans=True, span_min_length=4
        )
        for item in sst_dataset.spans:
            phrases.add(' '.join(item[0].split()[item[1]: item[2]]))
        print(len(sst_dataset), len(phrases))
    dataloader = DataLoader(
        sst_dataset, batch_size=64, shuffle=False, 
        collate_fn=sst_dataset.collate_fn
    )
    for batch in dataloader:
        pass
    # UD Test
    # Test 1: all UD as a joint dataset
    ud_dataset = UniversalDependenciesDataset(
        '../data/*/*/*conllu',
        '../data/universal-dependencies-1.2/tags.txt'
    )
    # Test 2: data loader
    ud_dataset = UniversalDependenciesDataset(
        '../data/*/*English/*conllu',
        '../data/universal-dependencies-1.2/tags.txt'
    )
    ud_dataloader = DataLoader(
        ud_dataset, batch_size=64, shuffle=False, 
        collate_fn=ud_dataset.collate_fn
    )
    # Test 3: tag ids
    for batch in ud_dataloader:
        ud_dataset.get_tag_ids(batch)
    from IPython import embed; embed(using=False)
