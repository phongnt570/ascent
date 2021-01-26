"""Executable script to run Bing search for subjects."""

import logging
from configparser import ConfigParser
from functools import partial
from multiprocessing import Pool
from typing import List
from urllib.parse import quote_plus

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

from filepath_handler import get_url_path, get_title_path, get_snippet_path, get_wiki_path, get_wiki_map_source_path, \
    get_article_dir
from helper.argument_parser import split_list_into_sublists
from helper.constants import MANUAL_WN2WP, EN_WIKIPEDIA_PREFIX, NEED_CRAWL
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

        # quick fix: integrate manual WordNet-Wikipedia mapping
        if subject.name() in MANUAL_WN2WP:
            wiki = EN_WIKIPEDIA_PREFIX + quote_plus(MANUAL_WN2WP[subject.name()]["wikipedia"])
            wiki_map_source = MANUAL_WN2WP[subject.name()]["source"]

            if get_url_path(subject).exists() and get_wiki_path(subject).exists():
                with get_wiki_path(subject).open() as f:
                    old_wiki = f.readline().strip()

                if wiki.lower() != old_wiki.lower():
                    logger.info(
                        f"Change of WordNet-Wikipedia mapping: current Wikipedia - {old_wiki} "
                        f"will be replaced by {wiki}")
                    with get_url_path(subject).open() as f:
                        urls = [line.strip() for line in f.readlines() if line.strip()]
                    with get_title_path(subject).open() as f:
                        titles = [line.strip() for line in f.readlines() if line.strip()]
                    with get_snippet_path(subject).open() as f:
                        snippets = [line.strip() for line in f.readlines() if line.strip()]

                    if wiki not in set(url.lower() for url in urls):
                        logger.info("New Wikipedia URL not in search results, append it.")
                        urls[-1] = wiki
                        titles[-1] = "{} - Wikipedia".format(wiki[(wiki.rindex("/") + 1):].replace("_", " "))
                        snippets[-1] = ""

                        # inform that crawled article (if exists) should be re-place
                        wiki_filepath = get_article_dir(subject) / "{}.txt".format(len(urls) - 1)
                        if wiki_filepath.exists():
                            with wiki_filepath.open("w+") as h:
                                h.write(NEED_CRAWL)
                    # write new data
                    write_results(subject, urls, titles, snippets, wiki, wiki_map_source)

        if not self._config["bing_search"].getboolean("overwrite"):
            if get_url_path(subject).exists():
                logger.info(f"{get_url_path(subject)} exists. Overwriting is not set. Search ignored.")
                return

        logger.info(f"Subject {subject.name()} - Bing search")

        results, wiki, wiki_map_source = search_for_subject(
            subject=subject,
            num_urls=self._config["bing_search"].getint("num_urls"),
            subscription_key=self._config["bing_search"]["subscription_key"],
            custom_config=self._config["bing_search"]["custom_config"],
            host=self._config["bing_search"]["host"],
            path=self._config["bing_search"]["path"]
        )

        urls, titles, snippets = zip(*results)
        write_results(subject, urls, titles, snippets, wiki, wiki_map_source)

        logger.info(f"Subject {subject.name()} - Collected URLs: {len(urls)}")
        logger.info(f"Subject {subject.name()} - Wikipedia URL: {wiki} - Source: {wiki_map_source}")


def write_results(subject: Synset, urls: List[str], titles: List[str], snippets: List[str], wiki: str,
                  wiki_map_source: str) -> None:
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

    with get_wiki_map_source_path(subject).open("w+") as f:
        f.write(wiki_map_source)
