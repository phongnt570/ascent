import json
from pathlib import Path
from typing import Union

import pandas as pd
from pandas import DataFrame
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

from filepath_handler import get_facet_labels_filepath

with get_facet_labels_filepath().open() as f:
    label2id = json.load(f)
id2label = {value: key for key, value in label2id.items()}


class FacetDataset(Dataset):
    def __init__(self, filename: Union[str, Path], tokenizer: RobertaTokenizer, maxlen: int = 32):
        self.df: DataFrame = pd.read_csv(filename)
        self.tokenizer: RobertaTokenizer = tokenizer
        self.maxlen: int = maxlen
        self.has_label: bool = 'facetType' in self.df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        subj = self.df.loc[index, 'subject']
        pred = self.df.loc[index, 'predicate']
        obj = self.df.loc[index, 'object']
        facet = self.df.loc[index, 'facetValue']

        text = " ".join([subj, "[pred]", pred, "[obj]", obj, "[facet]", facet])
        code = self.tokenizer.encode_plus(text, max_length=self.maxlen, pad_to_max_length=True, return_tensors='pt',
                                          truncation=True)

        result = [code['input_ids'][0], code['token_type_ids'][0], code['attention_mask'][0]]

        if self.has_label:
            result.append(label2id[self.df.loc[index, 'facetType']])

        return tuple(result)
