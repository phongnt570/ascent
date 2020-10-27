"""Executable script to compute documents' relevance to subjects."""

import logging
from configparser import ConfigParser
from multiprocessing import Pool
from typing import List

from nltk.corpus import wordnet as wn

from filepath_handler import get_relevant_scores_path
from helper.argument_parser import split_list_into_sublists
from pipeline.module_interface import Module
from retrieval.doc_filter import get_similarity_scores_of_all_articles

logger = logging.getLogger(__name__)


class ArticleFilteringModule(Module):
    def __init__(self, config: ConfigParser):
        super().__init__(config)
        self._name = "Filter irrelevant articles"

    def run(self, subject_list: List[str], **kwargs):
        # split subject list into `num_processes` equal batches
        batch_list = split_list_into_sublists(subject_list, self._config["filter"].getint("num_processes"))

        with Pool(len(batch_list)) as pool:
            pool.map(self.single_process, batch_list)

    def single_process(self, subject_list: List[str]):
        for subject in subject_list:
            try:
                self.single_subject_run(subject)
            except Exception as err:
                logger.critical(f"{subject}, {err}")

    def single_subject_run(self, subject: str):
        subject = wn.synset(subject)

        if not self._config["filter"].getboolean("overwrite"):
            if get_relevant_scores_path(subject).exists():
                logger.info(f"{get_relevant_scores_path(subject)} exists. Overwriting is not set. Filter ignored.")
                return

        logger.info(f'Subject {subject.name()} - Filter articles')

        scores = get_similarity_scores_of_all_articles(subject)

        with get_relevant_scores_path(subject).open("w+") as f:
            f.write("\n".join([str(score) for score in scores]))

        logger.info(f"Subject {subject.name()} - Filtering done")
