import json
import logging
from configparser import ConfigParser
from functools import partial
from multiprocessing import Pool
from typing import Tuple, List

from extraction.extractor import GENERAL_ASSERTION_KEY, SUBGROUP_ASSERTION_KEY, ASPECT_ASSERTION_KEY, WN_SYNSET_KEY
from facet_labeling.facet_labeling_factory import FacetLabelingFactory
from filepath_handler import get_facet_labeled_json_path, get_triple_clusters_json_path
from helper.argument_parser import split_subjects_to_gpus
from pipeline.module_interface import Module

logger = logging.getLogger(__name__)


class FacetLabelingModule(Module):
    def __init__(self, config: ConfigParser):
        super().__init__(config)
        self._name = "Label facets"

    def run(self, subject_list: List[str], **kwargs):
        subject_list = subject_list.copy()  # noqa

        # split subjects into gpus
        gpus, subject_batches = split_subjects_to_gpus(subject_list, gpus_arg=self._config["default"]["gpu"])

        logger.info("GPUs to be used: {}".format(gpus))

        # multi-processing
        processes = partial(self.single_gpu_run)
        with Pool(len(subject_batches)) as pool:
            pool.map(processes, zip(gpus, subject_batches))

    def single_gpu_run(self, gpu_id_and_subject_list: Tuple[int, List[str]]):
        cuda_device, subject_list = gpu_id_and_subject_list

        model_path = self._config["facet_labeling"]["model"]
        batch_size = self._config["facet_labeling"].getint("batch_size")

        logger.info(f'Loading facet labeler [GPU={cuda_device}]')
        labeler = FacetLabelingFactory(model_path=model_path,
                                       device=f"cuda:{cuda_device}" if cuda_device >= 0 else "cpu",
                                       batch_size=batch_size)

        data_list = []
        assertion_list = []

        for subject in subject_list:
            try:
                if not self._config["facet_labeling"].getboolean("overwrite"):
                    if get_facet_labeled_json_path(subject).exists():
                        logger.info(
                            f"{get_facet_labeled_json_path(subject)} exists. Overwriting is not set. "
                            f"Labeling ignored.")
                        continue

                with get_triple_clusters_json_path(subject).open() as f:
                    data = json.load(f)
                    data_list.append(data)
                    for subject_data in (
                            data[GENERAL_ASSERTION_KEY] + data[SUBGROUP_ASSERTION_KEY] + data[ASPECT_ASSERTION_KEY]):
                        assertion_list.extend(
                            [assertion for cluster in subject_data["clusters"] for assertion in cluster])
            except Exception as err:
                logger.critical(f"{subject}, {err}")

        labeler.label(assertion_list)

        logger.info('Updating and saving new JSON files...')
        for data in data_list:
            subject: str = data[WN_SYNSET_KEY]["synsetID"]
            with get_facet_labeled_json_path(subject).open("w+", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=False)

        logger.info('Job on GPU={} finished!'.format(cuda_device))
