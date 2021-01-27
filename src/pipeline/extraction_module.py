import logging
from configparser import ConfigParser
from functools import partial
from multiprocessing import Pool
from typing import List

import neuralcoref
from nltk.corpus import wordnet as wn

from extraction.extractor import single_run, get_concept_alias
from filepath_handler import get_kb_json_path
from helper.argument_parser import split_list_into_sublists
from pipeline.module_interface import Module
from static_resource import StaticResource

logger = logging.getLogger(__name__)


class ExtractionModule(Module):
    def __init__(self, config: ConfigParser):
        super().__init__(config)
        self._name = "Extract knowledge"

    def run(self, subject_list: List[str], **kwargs):
        # split subject list into `num_processes` equal batches
        batch_list = split_list_into_sublists(subject_list, self._config["extraction"].getint("num_processes"))

        processes = partial(self.single_process, doc_threshold=self._config["extraction"].getfloat("doc_threshold"))
        with Pool(len(batch_list)) as pool:
            pool.map(processes, batch_list)

    def single_process(self, subject_list: List[str], doc_threshold: float = 0.55) -> None:
        # get spacy
        spacy_nlp = StaticResource.nlp()
        neuralcoref.add_to_pipe(spacy_nlp)

        for subject in subject_list:
            try:
                concept = wn.synset(subject)

                if not self._config["extraction"].getboolean("overwrite"):
                    if get_kb_json_path(concept).exists():
                        logger.info(f"{get_kb_json_path(concept)} exists. Overwriting is not set. Extraction ignored.")
                        continue

                logger.info(f"Subject {concept.name()} - Extraction begins")
                single_run(concept=concept,
                           spacy_nlp=spacy_nlp,
                           doc_threshold=doc_threshold,
                           alias=get_concept_alias(concept))
                logger.info(f"Subject {concept.name()} - Extraction done")
            except Exception as err:
                logger.critical(f"{subject}, {err}")
