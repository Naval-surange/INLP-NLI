import json
import os
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer

from datasets.collate_functions import collate_to_max_length


class MNLIDataset(Dataset):

    def __init__(self, directory, prefix, bert_path, max_length: int = 512):
        super().__init__()
        self.max_length = max_length
        label_map = {"contradiction": 0, 'neutral': 1, "entailment": 2}
        with open(os.path.join(directory, 'multinli_1.0_' + prefix + '.jsonl'), 'r', encoding='utf8') as f:
            lines = f.readlines()
        self.result = []
        for line in lines:
            line_json = json.loads(line)
            if line_json['gold_label'] not in label_map:
                # print(line_json['gold_label'])
                continue
            self.result.append((line_json['sentence1'], line_json['sentence2'], label_map[line_json['gold_label']]))
        self.tokenizer = RobertaTokenizer.from_pretrained(bert_path)

    def __len__(self):
        return len(self.result)

    def __getitem__(self, idx):
        sentence_1, sentence_2, label = self.result[idx]
        # remove .
        if sentence_1.endswith("."):
            sentence_1 = sentence_1[:-1]
        if sentence_2.endswith("."):
            sentence_2 = sentence_2[:-1]
        sentence_1_input_ids = self.tokenizer.encode(sentence_1, add_special_tokens=False)
        sentence_2_input_ids = self.tokenizer.encode(sentence_2, add_special_tokens=False)
        input_ids = sentence_1_input_ids + [2] + sentence_2_input_ids
        if len(input_ids) > self.max_length - 2:
            input_ids = input_ids[:self.max_length - 2]
        # convert list to tensor
        length = torch.LongTensor([len(input_ids) + 2])
        input_ids = torch.LongTensor([0] + input_ids + [2])
        label = torch.LongTensor([label])
        return input_ids, label, length

