import json
import regex
from glob import glob
from torch.utils.data import Dataset


class Sentence(object):
    def __init__(self, conll_data):
        self.conll_data = conll_data
        self.ids, self.words, self.tags, self.deps, self.dep_labels = \
            self.extract_info(conll_data)
    
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

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

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


if __name__ == '__main__':
    # Test 1: all UD as a joint dataset
    dataset = UniversalDependenciesDataset(
        '../data/*/*/*conllu',
        '../data/universal-dependencies-1.2/tags.txt'
    )
    # Test 2: data loader
    from torch.utils.data import DataLoader
    dataset = UniversalDependenciesDataset(
        '../data/*/*English/*conllu',
        '../data/universal-dependencies-1.2/tags.txt'
    )
    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=False, 
        collate_fn=dataset.collate_fn
    )
    # Test 3: tag ids
    for batch in dataloader:
        dataset.get_tag_ids(batch)
    from IPython import embed; embed(using=False)
