"""Containing functions to extract all kinds of assertions given a subject."""
import csv
import json
import logging
from collections import Counter
from typing import List, Tuple, Dict, Set

from nltk.corpus.reader.wordnet import Synset
from spacy.language import Language
from spacy.tokens.doc import Doc

from experiment.rule_based_clustering import rule_based_clustering
from extraction.assertion import SimplifiedAssertion, Assertion
from extraction.extract_assertions import extract_general_and_subgroup_assertions, extract_subpart_assertions
from extraction.extract_terms import extract_subgroups, extract_subparts, Subgroup, Subpart
from extraction.stuffie import run_extraction
from filepath_handler import get_article_dir, get_kb_json_path, get_rule_based_clusters_filepath, \
    get_relevant_scores_path
from retrieval.doc_filter import get_wikipedia_url

CONCEPT_NAME_KEY = "concept_name"
SYNSET_KEY = "synset"
WIKIPEDIA_KEY = "wikipedia"
ALIAS_KEY = "alias"
SUBGROUP_KEY = "subgroups"
SUBPART_KEY = "subparts"
GENERAL_ASSERTION_KEY = "general_assertions"
SUBGROUP_ASSERTION_KEY = "subgroup_assertions"
SUBPART_ASSERTION_KEY = "subpart_assertions"
STATISTICS_KEY = "statistics"

logger = logging.getLogger(__name__)


def single_run(concept: Synset, spacy_nlp: Language, doc_threshold: float,
               do_clustering: bool = False, alias: List[str] = None, output_file: str = None):
    # get all relevant texts
    line_list, num_doc_retrieved, num_doc_retained = get_relevant_texts(concept, doc_threshold)

    # OpenIE
    logger.info(f"Subject {concept.name()} - Running SpaCy, NeuralCoref and StuffIE...")
    concept_name = get_concept_name(concept)
    if alias is None:
        alias = []
    subject_list = [concept_name]
    subject_list.extend(alias)
    doc_list, assertion_list, num_sentences = run_extraction(line_list, spacy_nlp, concept_name)

    extractions = []
    for subject in subject_list:
        extractions.append(
            extract(subject=subject, target_subject=concept_name, doc_list=doc_list, assertion_list=assertion_list,
                    spacy_nlp=spacy_nlp))

    subgroup_list, subpart_list, general_assertions, subgroup_assertions, subpart_assertions = merge(
        target_subject=concept_name, extractions=extractions, alias=alias)

    # rule-based clustering
    if do_clustering:
        logger.info(f"Subject {concept.name()} - Rule-based clustering...")
        run_rule_based_clustering(concept, general_assertions, subgroup_assertions, subpart_assertions)

    # statistics
    statistics = {
        "num_doc_retrieved": num_doc_retrieved,
        "num_doc_retained": num_doc_retained,
        "num_sentences": num_sentences,
        "num_extracted_assertions": len(assertion_list),
        "num_general_assertions": len(general_assertions),
        "num_subgroup_assertions": len(subgroup_assertions),
        "num_subpart_assertions": len(subpart_assertions),
        "num_subgroups": len(subgroup_list),
        "num_subparts": len(subpart_list),
        "num_facets": sum([len(a.facets) for a in (general_assertions + subgroup_assertions + subpart_assertions)]),
    }

    # print results to json file
    logger.info(f"Subject {concept.name()} - Printing results to JSON file...")
    json_obj = {
        CONCEPT_NAME_KEY: concept_name,
        SYNSET_KEY: concept.name(),
        WIKIPEDIA_KEY: get_wikipedia_url(concept),
        ALIAS_KEY: alias,
        SUBGROUP_KEY: [subgroup.to_dict() for subgroup in subgroup_list],
        SUBPART_KEY: [subpart.to_dict() for subpart in subpart_list],
        GENERAL_ASSERTION_KEY: [assertion.to_dict(simplifies_object=not do_clustering) for assertion in
                                general_assertions],
        SUBGROUP_ASSERTION_KEY: [assertion.to_dict(simplifies_object=not do_clustering) for assertion in
                                 subgroup_assertions],
        SUBPART_ASSERTION_KEY: [assertion.to_dict(simplifies_object=not do_clustering) for assertion in
                                subpart_assertions],
        STATISTICS_KEY: statistics,
    }
    if output_file is None:
        with get_kb_json_path(concept).open("w+", encoding="utf-8") as f:
            json.dump(json_obj, f, ensure_ascii=False, indent=2, sort_keys=False)
    else:
        with open(output_file, "w+") as f:
            json.dump(json_obj, f, ensure_ascii=False, indent=2, sort_keys=False)


def get_concept_name(concept: Synset) -> str:
    return concept.lemma_names()[0].replace("_", " ").replace("-", " ").strip().lower()


def get_concept_alias(subject: Synset) -> List[str]:
    r = [lemma.replace("_", " ").replace("-", " ").strip().lower() for lemma in subject.lemma_names()]
    return list(set([lemma for lemma in r[1:] if lemma != r[0]]))


def extract(subject: str, target_subject: str, doc_list: List[Doc], assertion_list: List[Assertion],
            spacy_nlp: Language) -> Dict[str, List]:
    if target_subject.endswith(" " + subject) or subject.endswith(" " + target_subject):
        subgroup_list, subpart_list = [], []
        general_assertions, subgroup_assertions = [], []
        subpart_assertions = []
    else:
        logger.info(f"Subject {target_subject} - alias: {subject}")
        # related terms
        subgroup_list = extract_subgroups(doc_list, subject, spacy_nlp)
        subpart_list = extract_subparts(doc_list, assertion_list, subject)
        # related assertions
        general_assertions, subgroup_assertions = extract_general_and_subgroup_assertions(assertion_list, subject,
                                                                                          subgroup_list)
        subpart_assertions = extract_subpart_assertions(assertion_list, subject, subgroup_list, subpart_list)

    return {
        'subgroup_list': subgroup_list,
        'subpart_list': subpart_list,
        'general_assertions': general_assertions,
        'subgroup_assertions': subgroup_assertions,
        'subpart_assertions': subpart_assertions,
    }


def merge(target_subject: str, extractions: List[Dict[str, List]], alias: List[str]) \
        -> Tuple[List[Subgroup], List[Subpart], List[SimplifiedAssertion],
                 List[SimplifiedAssertion], List[SimplifiedAssertion]]:
    alias_set: Set[str] = set(alias)

    # subgroups
    subgroup_list: List[Subgroup] = sorted(
        [subgroup for extraction in extractions for subgroup in extraction['subgroup_list'] if
         subgroup.name not in alias_set],
        key=lambda subgroup: subgroup.get_frequency(), reverse=True)

    # subparts
    subpart_list: List[Subpart] = [subpart for extraction in extractions for subpart in extraction['subpart_list']]
    subpart_list = sorted(merge_subparts(subpart_list), key=lambda subpart: subpart.get_frequency(), reverse=True)

    # general assertions
    general_assertions: List[SimplifiedAssertion] = [assertion for extraction in extractions for assertion in
                                                     extraction['general_assertions']]
    for assertion in general_assertions:
        assertion.subj = target_subject  # change back to target subject for consistency

    # subgroup assertions
    subgroup_assertions: List[SimplifiedAssertion] = [assertion for extraction in extractions for assertion in
                                                      extraction['subgroup_assertions']]
    # alias
    # e.g.: "boar" has alias "wild boar", then all "wild boar" assertions are general assertions
    # same as "panda" and "giant panda"
    g_idx: Set[int] = set()
    for idx, assertion in enumerate(subgroup_assertions):
        if assertion.subj in alias_set:
            assertion.subj = target_subject
            general_assertions.append(assertion)
            g_idx.add(idx)
    subgroup_assertions = [sga for idx, sga in enumerate(subgroup_assertions) if idx not in g_idx]

    # subpart assertions
    subpart_assertions: List[SimplifiedAssertion] = [assertion for extraction in extractions for assertion in
                                                     extraction['subpart_assertions']]

    return subgroup_list, subpart_list, general_assertions, subgroup_assertions, subpart_assertions


def merge_subparts(subpart_list: List[Subpart]) -> List[Subpart]:
    name2subpart: Dict[str, Subpart] = {}
    for subpart in subpart_list:
        if subpart.name in name2subpart:
            root_subpart = name2subpart[subpart.name]
            for chunk in subpart.chunk_list:
                root_subpart.add_chunk(chunk)
            for phrase, count in subpart.phrase_counter.items():
                root_subpart.add_phrase(phrase, count)
        else:
            name2subpart[subpart.name] = subpart
    return list(name2subpart.values())


def get_relevant_texts(subject: Synset, doc_threshold: float) -> Tuple[List[str], int, int]:
    """Get all lines from all relevant articles. Also return the number of retrieved documents and retained ones."""

    article_dir = get_article_dir(subject)
    rel_path = get_relevant_scores_path(subject)

    subject_name = get_concept_name(subject)

    with rel_path.open() as f:  # read file to get the ids of relevant articles
        scores = [float(line) for line in f if line.strip()]

    num_doc_retrieved = len(scores)
    line_list = []
    num_doc_retained = 0
    for doc_id, score in enumerate(scores):
        path = article_dir / "{}.txt".format(doc_id)
        try:
            with path.open() as f:
                lines = [line.strip() for line in f if line.strip()]

                if len(lines) > 500:  # ignore huge files
                    continue

                text = "\n".join(lines)

                if score >= doc_threshold or (len(text.split()) <= 200 and subject_name in text.lower()):
                    line_list.extend(lines)
                    num_doc_retained += 1
        except FileNotFoundError:
            logger.warning(f"Subject {subject.name()} - {path} does not exist!")
            continue

    return line_list, num_doc_retrieved, num_doc_retained


def run_rule_based_clustering(subject: Synset, general_assertions: List[SimplifiedAssertion],
                              subgroups_assertions: List[SimplifiedAssertion],
                              subpart_assertions: List[SimplifiedAssertion]) -> None:
    """Function to run rule based clustering for all kinds of assertion.
    This is only used to generate training data for BERT-based triple clustering method.
    Output is automatically written to csv files."""

    all_assertion_lists = [
        rule_based_clustering(general_assertions),
        rule_based_clustering(subgroups_assertions),
        rule_based_clustering(subpart_assertions)
    ]

    with get_rule_based_clusters_filepath(subject).open('w+') as f:
        writer = csv.DictWriter(f,
                                fieldnames=['cluster_id', 'subject_id', 'triple_id', 'subject', 'predicate', 'object',
                                            'object_root', 'count'])
        writer.writeheader()

        cluster_id, subject_id, triple_id = 0, 0, 0

        for assertion_list in all_assertion_lists:
            for _, assertions in assertion_list.items():
                for same_predicate_asst in assertions:
                    group = [simple_asst for same_object_asst in same_predicate_asst.same_object_assertion_list
                             for simple_asst in same_object_asst.simplified_assertion_list]
                    counter = Counter(group)
                    for asst, cnt in counter.most_common():
                        writer.writerow({
                            'cluster_id': cluster_id,
                            'subject_id': subject_id,
                            'triple_id': triple_id,
                            'subject': str(asst.subj),
                            'predicate': asst.pred,
                            'object': str(asst.obj),
                            'object_root': asst.obj.root.lemma_,
                            'count': cnt,
                        })
                        f.write('\n')
                        triple_id += 1
                    cluster_id += 1
                subject_id += 1
