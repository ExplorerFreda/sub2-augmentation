import copy
import dotdict
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class POSTagger(nn.Module):
    def __init__(
            self, model_name, tagset, device='cuda',
            hidden_dim=1024, dropout_p=0.2, fine_tune=False
        ):
        super(POSTagger, self).__init__()
        self.tagset = tagset
        self.device = device
        self.fine_tune = fine_tune
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = self.tokenizer.cls_token
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = self.tokenizer.sep_token
        self.pretrained_model = AutoModel.from_pretrained(
            model_name).to(device)
        self.max_length = \
            self.pretrained_model.config.max_position_embeddings - 2
        if not fine_tune:
            for p in self.pretrained_model.parameters():
                p.requires_grad = False
        
        self.layer_num = self.pretrained_model.config.num_hidden_layers + 1
        self.layer_weight = nn.Parameter(
            torch.zeros(1, self.layer_num).to(device)
        )
        self.feature_dim = self.pretrained_model.config.hidden_size * 2
        layers = [
            nn.Dropout(p=dropout_p),
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
        lbs = torch.tensor(self.__pad__(lbs, -1)).long().to(self.device)
        rbs = torch.tensor(self.__pad__(rbs, -1)).long().to(self.device)
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

    def forward(self, ids, attention_masks, lbs, rbs):
        layerwise_output = list()
        for sid in range(0, ids.shape[1], self.max_length):
            sub_ids = ids[:, sid:sid+self.max_length]
            sub_attn_masks = attention_masks[:, sid:sid+self.max_length]
            _, _, sub_layerwise_output = self.pretrained_model(
                sub_ids, sub_attn_masks, output_hidden_states=True
            )
            sub_layerwise_output = torch.cat(
                [x.unsqueeze(-1) for x in sub_layerwise_output], dim=-1
            )
            layerwise_output.append(sub_layerwise_output)
        layerwise_output = torch.cat(layerwise_output, dim=1)
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

    @staticmethod
    def forward_batch(model, batch):
        abs_model = model if isinstance(model, POSTagger) else model.module
        ids, attention_masks, lbs, rbs = abs_model.__tokenize__(batch)
        logits = model(ids, attention_masks, lbs, rbs)
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