import collections
import copy
import random
from tqdm import tqdm


class Augmenter(object):
    def __init__(self, dataset):
        self.dataset = copy.deepcopy(dataset)


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
        

if __name__ == "__main__":
    from data import UniversalDependenciesDataset
    dataset = UniversalDependenciesDataset(
        '../data/*/*/en*train*conllu',
        '../data/universal-dependencies-1.2/tags.txt'
    )
    augmenter = POSTagAugmenter(dataset)
    augmented_dataset = augmenter.augment()
    from IPython import embed; embed(using=False)