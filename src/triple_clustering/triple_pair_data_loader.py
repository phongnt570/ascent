"""The data loader class used for training BERT-based models towards triple clustering."""
from pathlib import Path
from typing import Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer


class TriplePairDataset(Dataset):
    def __init__(self, filename: Union[str, Path], tokenizer: RobertaTokenizer, maxlen: int = 32,
                 do_train: bool = False):
        super().__init__()
        self.df = pd.read_csv(filename)
        self.tokenizer = tokenizer
        self.do_train = do_train
        self.has_label = 'label' in self.df
        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        po1 = (self.df.loc[index, 'predicate_1'], self.df.loc[index, 'object_1'])
        po2 = (self.df.loc[index, 'predicate_2'], self.df.loc[index, 'object_2'])

        # Randomly swap 2 triples when training
        if self.do_train:
            if torch.rand(1)[0] >= 0.5:
                tmp = po1
                po1 = po2
                po2 = tmp

        # Pre-processing the text to be suitable for BERT
        text1 = ['[subj]', po1[0], '[u-sep]', po1[1]]
        text2 = ['[subj]', po2[0], '[u-sep]', po2[1]]
        text1 = ' '.join(text1)
        text2 = ' '.join(text2)
        code = self.tokenizer.encode_plus(text1, text2, max_length=self.maxlen, padding="max_length",
                                          return_tensors='pt',)

        result = [code['input_ids'][0], code['token_type_ids'][0], code['attention_mask'][0]]

        if self.has_label:
            result.append(self.df.loc[index, 'label'])

        return result
