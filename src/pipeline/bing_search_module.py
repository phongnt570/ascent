"""Executable script to run Bing search for subjects."""

import logging
from configparser import ConfigParser
from functools import partial
from multiprocessing import Pool
from typing import List

from nltk.corpus import wordnet as wn

from filepath_handler import get_url_path, get_title_path, get_snippet_path, get_wiki_path
from helper.argument_parser import split_list_into_sublists
from pipeline.module_interface import Module
from retrieval.bing_search import search_for_subject

logger = logging.getLogger(__name__)


class BingSearchModule(Module):
    def __init__(self, config: ConfigParser):
        super().__init__(config)
        self._name = "Bing Search"

    def run(self, subject_list: List[str], **kwargs):
        # split subject list into `num_processes` equal batches
        batch_list = split_list_into_sublists(subject_list, self._config["bing_search"].getint("num_processes"))

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

        if not self._config["bing_search"].getboolean("overwrite"):
            if get_url_path(subject).exists():
                logger.info(f"{get_url_path(subject)} exists. Overwriting is not set. Search ignored.")
                return

        logger.info(f"Subject {subject.name()} - Bing search")

        results, wiki = search_for_subject(subject=subject,
                                           num_urls=self._config["bing_search"].getint("num_urls"),
                                           subscription_key=self._config["bing_search"][
                                               "subscription_key"],
                                           custom_config=self._config["bing_search"]["custom_config"],
                                           host=self._config["bing_search"]["host"],
                                           path=self._config["bing_search"]["path"])

        urls, titles, snippets = zip(*results)

        with get_url_path(subject).open("w+") as f:
            for url in urls:
                f.write(url + "\n")

        with get_title_path(subject).open("w+") as f:
            for title in titles:
                f.write(title + "\n")

        with get_snippet_path(subject).open("w+") as f:
            for snippet in snippets:
                f.write(snippet + "\n")

        with get_wiki_path(subject).open("w+") as f:
            f.write(wiki)

        logger.info(f"Subject {subject.name()} - Collected URLs: {len(urls)}")
        logger.info(f"Subject {subject.name()} - Wikipedia URL: {wiki}")
