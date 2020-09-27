import copy
import dotdict
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class POSTagger(nn.Module):
    def __init__(
            self, model_name, tagset, device='cuda',
            hidden_dim=1024, fine_tune=False
        ):
        super(POSTagger, self).__init__()
        self.tagset = tagset
        self.device = device
        self.fine_tune = fine_tune
        self.hidden_dim = hidden_dim

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pretrained_model = AutoModel.from_pretrained(
            model_name).to(device)
        if not fine_tune:
            for p in self.pretrained_model.parameters():
                p.requires_grad = False
        
        self.layer_num = self.pretrained_model.config.num_hidden_layers + 1
        self.layer_weight = nn.Parameter(
            torch.zeros(1, self.layer_num).to(device)
        )
        self.feature_dim = self.pretrained_model.config.hidden_size * 2
        layers = [
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, len(tagset))
        ]
        self.classifier = nn.Sequential(*layers).to(device)
    
    def __tokenize_sentence__(self, sentence):
        subwords, lbs, rbs = list(), list(), list()
        subwords.append(self.tokenizer.bos_token)
        for word in sentence.words:
            lbs.append(len(subwords))
            tokens = self.tokenizer.tokenize(word)
            subwords.extend(tokens)
            rbs.append(len(subwords) - 1)
        subwords.append(self.tokenizer.eos_token)
        attention_mask = [1] * len(subwords)
        return subwords, attention_mask, lbs, rbs

    def __tokenize__(self, sentences):
        tokenized_sentences = [
            self.__tokenize_sentence__(x) for x in sentences
        ]
        subwords, attention_masks, lbs, rbs = zip(*tokenized_sentences)
        subwords = self.__pad__(subwords, self.tokenizer.pad_token)
        attention_masks = self.__pad__(attention_masks, 0)
        lbs = self.__pad__(lbs, -1)
        rbs = self.__pad__(rbs, -1)
        ids = torch.tensor([
            self.tokenizer.convert_tokens_to_ids(x) for x in subwords
        ]).to(self.device)
        attention_masks = torch.tensor(attention_masks).to(self.device)
        return ids, attention_masks, lbs, rbs

    @staticmethod
    def __pad__(data, padding_token):
        max_length = max(len(x) for x in data)
        padded_data = list(data)
        for i in range(len(padded_data)):
            padded_data[i] = list(padded_data[i]) + [padding_token] * (
                max_length - len(padded_data[i])
            )
        return padded_data

    def forward(self, batch):
        ids, attention_masks, lbs, rbs = self.__tokenize__(batch)
        _, _, layerwise_output = self.pretrained_model(
            ids, attention_masks, output_hidden_states=True
        )
        layerwise_output = torch.cat(
            [x.unsqueeze(-1) for x in layerwise_output], dim=-1
        )
        subword_features = (
            self.layer_weight.softmax(dim=1) * layerwise_output
        ).sum(-1)
        left_features = subword_features[
            torch.arange(len(lbs)).unsqueeze(-1), lbs        
        ]
        right_features = subword_features[
            torch.arange(len(rbs)).unsqueeze(-1), rbs
        ]
        word_features = torch.cat([left_features, right_features], dim=-1)
        logits = self.classifier(word_features)
        return logits


if __name__ == '__main__':
    # Test 1: sentence batchify
    from data import UniversalDependenciesDataset
    from torch.utils.data import DataLoader
    dataset = UniversalDependenciesDataset(
        '../data/*/*Hebrew/*conllu',
        '../data/universal-dependencies-1.2/tags.txt'
    )
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=False, 
        collate_fn=dataset.collate_fn
    )
    tagger = POSTagger(
        'xlm-roberta-large', 
        dataset.tag2id
    )
    for batch in dataloader:
        tagger(batch)
        break
    from IPython import embed; embed(using=False)