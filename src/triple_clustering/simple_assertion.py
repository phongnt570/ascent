"""Simple data structure for assertion that contains pure strings and can be loaded from json files."""
import logging
from typing import List, Dict, Any

import numpy as np
from spacy.tokens.token import Token

from static_resource import StaticResource

logger = logging.getLogger(__name__)


class SimpleFacet(object):
    def __init__(self, data: Dict[str, str]):
        super().__init__()
        self.connector: str = data['connector']
        self.statement: str = data['statement']
        self.label: str = data.get('label', None)

    def __hash__(self):
        return hash(self.get_facet_str().lower())

    def __eq__(self, value):
        return self.get_facet_str().lower() == value.get_facet_str().lower()

    def get_facet_str(self) -> str:
        text = ""
        if self.connector is not None:
            text += self.connector
        if self.statement is not None:
            text += " " + self.statement
        return text.strip()

    def get_head_word(self) -> Token:
        tokens = StaticResource.nlp().tokenizer(self.get_facet_str())
        return tokens[-1]

    def get_vector(self) -> np.ndarray:
        array = [word.vector for word in StaticResource.nlp().tokenizer(self.get_facet_str())]
        return np.mean(np.array(array), axis=0)

    def to_dict(self) -> dict:
        return {
            'connector': self.connector,
            'statement': self.statement,
            'label': self.label,
        }


class SimpleAssertion(object):
    def __init__(self, data: Dict[str, Any]):
        super().__init__()
        self.subj: str = data['subject']
        self.pred: str = data['predicate']
        self.obj: str = data['object']
        self.facets: List[SimpleFacet] = [SimpleFacet(facet) for facet in data['facets']]
        self.source: str = data['source']

    def get_triple_str(self) -> str:
        return self.subj + " " + self.pred + " " + self.obj

    def __hash__(self):
        return hash(self.get_triple_str().lower())

    def __eq__(self, value):
        return self.get_triple_str().lower() == value.get_triple_str().lower()

    def get_vector(self) -> np.ndarray:
        """Vector for (predicate + object)."""
        array = [word.vector for word in StaticResource.nlp().tokenizer(self.pred)]
        array.extend([word.vector for word in StaticResource.nlp().tokenizer(self.obj)])
        return np.mean(np.array(array), axis=0)

    def get_object_vector(self) -> np.ndarray:
        """Vector for object only, stop words and punctuations discarded."""
        array = [StaticResource.nlp().vocab[word.lemma_].vector
                 for word in StaticResource.nlp().tokenizer(self.obj) if not (word.is_stop or word.is_punct)
                 ]
        if len(array) == 0:
            return np.zeros(StaticResource.nlp().vocab.vectors_length)
        return np.mean(np.array(array), axis=0)

    def get_obj_tokens(self) -> List[Token]:
        return StaticResource.nlp().tokenizer(self.obj)

    def get_obj_head_word(self) -> str:
        return self.get_obj_tokens()[-1].lemma_.lower()

    def to_dict(self) -> dict:
        return {
            'subject': self.subj,
            'predicate': self.pred,
            'object': self.obj,
            'facets': [
                facet.to_dict() for facet in self.facets
            ],
            'source': self.source
        }

    def get_simplified_object(self) -> str:
        return self.obj.replace(" ", "").replace("-", "").lower()
