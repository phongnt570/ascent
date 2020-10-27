import logging
from configparser import ConfigParser
from multiprocessing import Pool
from typing import List

from facet_grouping.grouping import group_for_one_subject
from filepath_handler import get_final_kb_json_path
from helper.argument_parser import split_list_into_sublists
from pipeline.module_interface import Module
from nltk.corpus import wordnet as wn

logger = logging.getLogger(__name__)


class FacetGroupingModule(Module):
    def __init__(self, config: ConfigParser):
        super().__init__(config)
        self._name = "Group similar facets"

    def run(self, subject_list: List[str], **kwargs):
        # split subject list into `num_processes` equal batches
        batch_list = split_list_into_sublists(subject_list, self._config["facet_grouping"].getint("num_processes"))

        logger.info("Clustering similar facets...")

        with Pool(len(batch_list)) as pool:
            pool.map(self.single_process, batch_list)

        logger.info("Done!")

    def single_process(self, subject_list: List[str]):
        for subject in subject_list:
            try:
                concept = wn.synset(subject)

                if not self._config["facet_grouping"].getboolean("overwrite"):
                    if get_final_kb_json_path(concept).exists():
                        logger.info(
                            f"{get_final_kb_json_path(concept)} exists. Overwriting is not set. Grouping ignored.")
                        continue

                group_for_one_subject(concept)
            except Exception as err:
                logger.critical(f"{subject}, {err}")
