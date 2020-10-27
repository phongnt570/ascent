import logging

import spacy
from spacy.language import Language

from helper.constants import SPACY_MODEL_NAME

logger = logging.getLogger(__name__)


class StaticResource(object):
    _nlp = None

    @staticmethod
    def load_spacy():
        logger.info('Loading Spacy model...')
        StaticResource._nlp = spacy.load(SPACY_MODEL_NAME)

    @staticmethod
    def nlp() -> Language:
        if StaticResource._nlp is None:
            StaticResource.load_spacy()
        return StaticResource._nlp
