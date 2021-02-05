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
        facet_list, source_list, sentence_list = [], [], []

        for assertion in assertion_list:
            if len(assertion["facets"]) != len(assertion["source"]["facets_matches"]):
                logger.warning(f"Facets and sources do not match: "
                               f"{assertion['subject']} ; {assertion['predicate']} ; {assertion['object']}. "
                               f"Created fake positions (Nones).")
                assertion["source"]["facets_matches"] = [{"matches": {"start": None, "end": None, "start_char": None,
                                                                      "end_char": None}}] * len(assertion["facets"])
            for facet, source in zip(assertion["facets"], assertion["source"]["facets_matches"]):
                facet_list.append(facet)
                source_list.append(source)
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
                input_batch = self.tokenizer.batch_encode_plus(
                    batch,
                    return_tensors="pt",
                    padding="max_length",
                    truncation='longest_first',
                    max_length=32
                )

                if self.device != "cpu":
                    for k in input_batch:
                        input_batch[k] = input_batch[k].to(self.device)

                logits = self.model(**input_batch)[0].detach().cpu().numpy()
                label_list.extend([self.model.config.id2label[ind] for ind in np.argmax(logits, axis=1)])

                batch_cnt += 1

        for facet, source, label in zip(facet_list, source_list, label_list):
            facet["label"] = label
            source["label"] = label


def prepare(assertion: Dict[str, Union[str, Dict[str, str]]], facet: Dict[str, str]) -> str:
    return " ".join([
        assertion["subject"],
        "[pred]",
        assertion["predicate"],
        "[obj]",
        assertion["object"],
        "[facet]",
        get_facet_text(facet)
    ])


def get_facet_text(facet: Dict[str, str]) -> str:
    text = ""
    if facet['connector'] is not None:
        text += facet['connector']
    if facet['statement'] is not None:
        text += " " + facet['statement']
    return text.strip()
