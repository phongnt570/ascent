import logging
from typing import List, Dict, Union, Any

import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

logger = logging.getLogger(__name__)


class FacetLabelingFactory(object):
    def __init__(self, model_path: str, device, batch_size: int):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaForSequenceClassification.from_pretrained(model_path)
        self.device = device
        self.model.eval()
        if self.device != "cpu":
            self.model.to(self.device)
        self.batch_size = batch_size

    def label(self, assertion_list: List[Dict[str, Any]]) -> None:
        facet_list, sentence_list = [], []

        for assertion in assertion_list:
            for facet in assertion["facets"]:
                facet_list.append(facet)
                sentence_list.append(prepare(assertion, facet))

        logger.info('Labeling {} facets'.format(len(facet_list)))

        num_batches = int(len(facet_list) / self.batch_size) + 1
        logger.info('Split into [{} batches]...'.format(num_batches))

        label_list = []
        with torch.no_grad():
            batch_cnt = 0
            for i in range(0, len(sentence_list), self.batch_size):
                logger.info('Batch {} / {}'.format(batch_cnt + 1, num_batches))

                batch = sentence_list[i:(i + self.batch_size)]
                input_batch = self.tokenizer.batch_encode_plus(batch, return_tensors="pt", pad_to_max_length=True,
                                                               max_length=32)

                if self.device != "cpu":
                    for k in input_batch:
                        input_batch[k] = input_batch[k].to(self.device)

                logits = self.model(**input_batch)[0].detach().cpu().numpy()
                label_list.extend([self.model.config.id2label[ind] for ind in np.argmax(logits, axis=1)])

                batch_cnt += 1

        for facet, label in zip(facet_list, label_list):
            facet["label"] = label


def prepare(assertion: Dict[str, Union[str, Dict[str, str]]], facet: Dict[str, str]) -> str:
    return " ".join(
        [assertion["subject"], "[pred]", assertion["predicate"], "[obj]", assertion["object"], "[facet]",
         get_facet_text(facet)])


def get_facet_text(facet: Dict[str, str]) -> str:
    text = ""
    if facet['connector'] is not None:
        text += facet['connector']
    if facet['statement'] is not None:
        text += " " + facet['statement']
    return text.strip()
