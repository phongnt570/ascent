"""Script to cluster triples using BERT-based method."""

import json
import logging
from configparser import ConfigParser
from functools import partial
from multiprocessing import Pool
from typing import List, Dict, Tuple

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

from extraction.extractor import GENERAL_ASSERTION_KEY, SUBGROUP_ASSERTION_KEY, ASPECT_ASSERTION_KEY, STATISTICS_KEY, \
    PROMINENT_LEMMA_KEY
from filepath_handler import get_kb_json_path, get_triple_clusters_json_path
from helper.argument_parser import split_subjects_to_gpus
from pipeline.module_interface import Module
from triple_clustering.simple_assertion import SimpleAssertion
from triple_clustering.triple_clustering_factory import TripleClusteringFactory

logger = logging.getLogger(__name__)


class TripleClusteringModule(Module):
    def __init__(self, config: ConfigParser):
        super().__init__(config)
        self._name = "Cluster similar triples"

    def run(self, subject_list: List[str], **kwargs):
        subject_list = subject_list.copy()  # noqa

        # split subjects into gpus
        gpus, subject_batches = split_subjects_to_gpus(subject_list, gpus_arg=self._config["default"]["gpu"])

        logger.info("GPUs to be used: {}".format(gpus))

        # multi-processing
        processes = partial(self.single_gpu_run)
        with Pool(len(subject_batches)) as pool:
            pool.map(processes, zip(gpus, subject_batches))

    def single_gpu_run(self, cuda_device_and_subject_list: Tuple[int, List[str]]):
        cuda_device, subject_list = cuda_device_and_subject_list

        model_path = self._config["triple_clustering"]["model"]
        distance_threshold = self._config["triple_clustering"].getfloat("threshold")
        batch_size = self._config["triple_clustering"].getint("batch_size")

        # loading the model
        logger.info(f"Loading classification model [GPU={cuda_device}]")
        factory = TripleClusteringFactory(model_path=model_path,
                                          device=f"cuda:{cuda_device}" if cuda_device >= 0 else "cpu",
                                          distance_threshold=distance_threshold,
                                          batch_size=batch_size)

        for subject in subject_list:
            try:
                concept = wn.synset(subject)

                if not self._config["triple_clustering"].getboolean("overwrite"):
                    if get_triple_clusters_json_path(concept).exists():
                        logger.info(
                            f"{get_triple_clusters_json_path(concept)} exists. Overwriting is not set. "
                            f"Clustering ignored.")
                        continue

                logger.info(f"Subject {concept.name()} - Run triple clustering")
                data = run_triple_clustering_for_subject(concept, factory)
                with get_triple_clusters_json_path(concept).open("w+", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, sort_keys=False)
                logger.info(f"Subject {concept.name()} - Triple clustering is done")
            except Exception as err:
                logger.critical(f"{subject}, {err}")


def run_triple_clustering_for_subject(subject: Synset, factory: TripleClusteringFactory, input_file: str = None) \
        -> dict:
    """Single clustering run for one subject."""

    logger.info('Reading assertions from JSON file')
    if input_file is None:
        data = read_original_assertions_from_json(subject)
    else:
        with open(input_file) as f:
            data = json.load(f)

    logger.info(f'Subject {subject.name()} - Clustering general triples')
    general_assertion_list = get_assertion_list(data, key=GENERAL_ASSERTION_KEY)
    general_clusters = bert_based_triple_clustering(general_assertion_list, factory,
                                                    prominent_lemma=data[PROMINENT_LEMMA_KEY])

    logger.info(f'Subject {subject.name()} - Clustering subgroup triples')
    sub_group_assertion_list = get_assertion_list(data, key=SUBGROUP_ASSERTION_KEY)
    sub_group_clusters = bert_based_triple_clustering(sub_group_assertion_list, factory)

    logger.info(f'Subject {subject.name()} - Clustering subpart triples')
    sub_part_assertion_list = get_assertion_list(data, key=ASPECT_ASSERTION_KEY)
    sub_part_clusters = bert_based_triple_clustering(sub_part_assertion_list, factory)

    # update json data
    data[GENERAL_ASSERTION_KEY] = build_dict_from_cluster_list(general_clusters)
    data[SUBGROUP_ASSERTION_KEY] = build_dict_from_cluster_list(sub_group_clusters)
    data[ASPECT_ASSERTION_KEY] = build_dict_from_cluster_list(sub_part_clusters)

    # update statistics
    data[STATISTICS_KEY].update({
        "num_canonical_general_assertions": sum([len(name["clusters"]) for name in data[GENERAL_ASSERTION_KEY]]),
        "num_canonical_subgroup_assertions": sum([len(name["clusters"]) for name in data[SUBGROUP_ASSERTION_KEY]]),
        "num_canonical_aspect_assertions": sum([len(name["clusters"]) for name in data[ASPECT_ASSERTION_KEY]]),
    })

    # update prominent lemma:
    # if data[GENERAL_ASSERTION_KEY]:
    #     data[PROMINENT_LEMMA_KEY] = data[GENERAL_ASSERTION_KEY][0]["subject"]
    # else:
    #     data[PROMINENT_LEMMA_KEY] = get_concept_name(subject)

    return data


def build_dict_from_cluster_list(subject2clusters: Dict[str, List[List[SimpleAssertion]]]) -> List[dict]:
    """Build dict object given a cluster list, used for saving results to json file."""

    data = []
    for subject, clusters in sorted(subject2clusters.items(), key=lambda sc: -len(sc[1])):
        data.append({
            "subject": subject,
            "clusters": [
                [assertion.to_dict() for assertion in cluster]
                for cluster in clusters
            ]
        })

    return data


def get_assertion_list(data: dict, key: str):
    """Get the kind of assertion list (i.e., general, subgroup or subpart assertions)."""

    return [SimpleAssertion(a) for a in data[key]]


def read_original_assertions_from_json(subject: Synset) -> dict:
    """Getting assertions from json file printed in the extraction phase."""

    with get_kb_json_path(subject).open() as f:
        data = json.load(f)
    return data


def bert_based_triple_clustering(assertion_list: List[SimpleAssertion], factory: TripleClusteringFactory,
                                 prominent_lemma: str = None) -> Dict[str, List[List[SimpleAssertion]]]:
    """Main function for BERT-based approach to triple clustering."""

    # group same subjects
    if not prominent_lemma:
        subject2assertion_list = same_subject_grouping(assertion_list)
    else:  # pick the lemma with most assertions as representative
        subject2assertion_list = {prominent_lemma: assertion_list}

    # clustering
    subject2clusters = {}
    for subject in subject2assertion_list:
        same_subject_assertion_list = subject2assertion_list[subject]
        clusters = factory.cluster(same_subject_assertion_list)
        clusters = sorted(clusters, key=lambda x: -len(x))
        subject2clusters[subject] = clusters

    return subject2clusters


def same_subject_grouping(assertion_list: List[SimpleAssertion]) -> Dict[str, List[SimpleAssertion]]:
    """Group assertion list by subject (exact string matching)."""

    subject2assertion_list = {}
    for assertion in assertion_list:
        subj = assertion.subj
        if subj in subject2assertion_list:
            subject2assertion_list[subj].append(assertion)
        else:
            subject2assertion_list[subj] = [assertion]
    return subject2assertion_list
