"""Executable script to crawl relevant texts for subjects."""

import logging
from configparser import ConfigParser
from functools import partial
from multiprocessing.dummy import Pool
from time import time
from typing import List

from nltk.corpus import wordnet as wn

from filepath_handler import get_article_dir
from helper.argument_parser import split_list_into_sublists
from pipeline.module_interface import Module
from retrieval.grab_article import grab_text, get_urls

logger = logging.getLogger(__name__)


class ArticleGrabModule(Module):
    def __init__(self, config: ConfigParser):
        super().__init__(config)
        self._name = "Crawl articles"

    def run(self, subject_list: List[str], **kwargs):
        # split subject list into `num_processes` equal batches
        batch_list = split_list_into_sublists(subject_list, self._config["article_grab"].getint("num_crawlers"))

        processes = partial(self.single_process)
        with Pool(len(batch_list)) as pool:
            pool.map(processes, batch_list)

    def single_process(self, subject_list: List[str]):
        for subject in subject_list:
            try:
                self.single_subject_run(subject)
            except Exception as err:
                logger.critical(f"{subject}, {err}")

    def single_subject_run(self, subject: str):
        subject = wn.synset(subject)

        if not self._config["article_grab"].getboolean("overwrite"):
            if any((get_article_dir(subject) / "{}.txt".format(i)).exists() for i in range(500)):
                logger.info(f"Subject {subject.name()} - Articles exist. Overwriting is not set. Crawling ignored.")
                return

        logger.info(f"Subject {subject.name()} - Crawl relevant articles")
        ts = time()

        grab = partial(grab_text, subject)
        urls = get_urls(subject)

        with Pool(self._config["article_grab"].getint("processes_per_crawler")) as p:
            p.map(grab, enumerate(urls))

        logger.info(f"Subject {subject.name()} - Took {time() - ts} seconds.")
