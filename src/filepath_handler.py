import argparse
import os
from configparser import ConfigParser
from pathlib import Path
from typing import Union

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset


class UniversalFilePathHandler(object):
    # default
    OUTPUT_DIR = Path("../output")
    RES_DIR = Path("../resources")

    @staticmethod
    def set_up(config: ConfigParser):
        UniversalFilePathHandler.OUTPUT_DIR = Path(config["default"]["out_dir"])
        UniversalFilePathHandler.RES_DIR = Path(config["default"]["res_dir"])

        try:
            if not UniversalFilePathHandler.OUTPUT_DIR.exists():
                UniversalFilePathHandler.OUTPUT_DIR.mkdir()
        except FileExistsError:
            pass


# output
def get_url_dir():
    directory = UniversalFilePathHandler.OUTPUT_DIR / "url"
    try:
        if not directory.exists():
            directory.mkdir()
    except FileExistsError:
        pass

    return directory


def get_art_dir():
    directory = UniversalFilePathHandler.OUTPUT_DIR / "article"
    try:
        if not directory.exists():
            directory.mkdir()
    except FileExistsError:
        pass

    return directory


def get_kb_parent_dir():
    directory = UniversalFilePathHandler.OUTPUT_DIR / "kb"
    try:
        if not directory.exists():
            directory.mkdir()
    except FileExistsError:
        pass

    return directory


def get_rel_dir():
    directory = UniversalFilePathHandler.OUTPUT_DIR / "relevant"
    try:
        if not directory.exists():
            directory.mkdir()
    except FileExistsError:
        pass

    return directory


def get_okb_dir():
    directory = UniversalFilePathHandler.OUTPUT_DIR / "other_kb"
    try:
        if not directory.exists():
            directory.mkdir()
    except FileExistsError:
        pass

    return directory


def get_title_dir():
    directory = UniversalFilePathHandler.OUTPUT_DIR / "title"
    try:
        if not directory.exists():
            directory.mkdir()
    except FileExistsError:
        pass

    return directory


def get_snippet_dir():
    directory = UniversalFilePathHandler.OUTPUT_DIR / "snippet"
    try:
        if not directory.exists():
            directory.mkdir()
    except FileExistsError:
        pass

    return directory


def get_wiki_dir():
    directory = UniversalFilePathHandler.OUTPUT_DIR / "wikipedia"
    try:
        if not directory.exists():
            directory.mkdir()
    except FileExistsError:
        pass

    return directory


# resource
def get_misc_dir():
    return UniversalFilePathHandler.RES_DIR / "misc"


def get_other_cskb_res_dir():
    return UniversalFilePathHandler.RES_DIR / "other-cskbs"


def get_canonical_name(synset: Union[str, Synset]):
    if isinstance(synset, str):
        synset = wn.synset(synset)
    return synset.name()


def get_title_path(subject):
    subject = get_canonical_name(subject)
    return get_title_dir() / "{}.txt".format(subject)


def get_snippet_path(subject):
    subject = get_canonical_name(subject)
    return get_snippet_dir() / "{}.txt".format(subject)


def get_url_path(subject):
    subject = get_canonical_name(subject)
    return get_url_dir() / "{}.txt".format(subject)


def get_wiki_path(subject):
    subject = get_canonical_name(subject)
    return get_wiki_dir() / "{}.txt".format(subject)


def get_wiki_map_source_path(subject):
    subject = get_canonical_name(subject)
    return get_wiki_dir() / "{}.src.txt".format(subject)


def get_article_dir(subject):
    subject = get_canonical_name(subject)
    new_art_dir = get_art_dir() / subject
    try:
        if not new_art_dir.exists():
            new_art_dir.mkdir()
    except FileExistsError:
        pass
    return new_art_dir


def get_kb_dir(subject):
    subject = get_canonical_name(subject)
    new_kb_dir = get_kb_parent_dir() / subject
    if not new_kb_dir.exists():
        new_kb_dir.mkdir()
    return new_kb_dir


def get_relevant_scores_path(subject):
    subject = get_canonical_name(subject)
    return get_rel_dir() / "{}.txt".format(subject)


def get_kb_json_path(subject):
    return get_kb_dir(subject) / "assertions.json"


def get_triple_clusters_json_path(subject):
    return get_kb_dir(subject) / "triple_clusters.json"


def get_srl_facet_labeled_json_path(subject):
    return get_kb_dir(subject) / "srl_facet_labeled.json"


def get_facet_labeled_json_path(subject):
    return get_kb_dir(subject) / "facet_labeled.json"


def get_final_kb_json_path(subject):
    return get_kb_dir(subject) / "final.json"


def get_final_kb_csv_path(subject):
    return get_kb_dir(subject) / "final.csv"


def get_final_output_csv_path():
    return get_kb_parent_dir() / "descent.csv"


def get_rule_based_clusters_filepath(subject):
    return get_kb_dir(subject) / "rule_based_clusters.csv"


def get_moby_filepath():
    return get_misc_dir() / "moby.txt"


def get_ccn_csk_relations_filepath():
    return get_other_cskb_res_dir() / "conceptnet_csk_r.txt"


def get_conceptnet_filepath():
    return get_other_cskb_res_dir() / "conceptnet.csv"


def get_tuplekb_filepath():
    return get_other_cskb_res_dir() / 'tuplekb.tsv'


def get_quasimodo_filepath():
    return get_other_cskb_res_dir() / 'quasimodo.tsv'


def get_other_kb_json_filepath(subject):
    return get_okb_dir() / "{}.json".format(get_canonical_name(subject))


def get_synonyms_filepath():
    return get_misc_dir() / "synonyms.tsv"


def get_pronouns_filepath():
    return get_misc_dir() / "pronouns.txt"


def get_ignored_pronouns_in_obj_filepath():
    return get_misc_dir() / "ignored_pronouns_in_object.txt"


def get_modal_verbs_filepath():
    return get_misc_dir() / "modal_verbs.txt"


def get_ignored_one_word_objects_filepath():
    return get_misc_dir() / "ignored_one_word_objects.txt"


def get_ignored_predicates_filepath():
    return get_misc_dir() / "ignored_predicates.txt"


def get_conjugate_exceptions_filepath():
    return get_misc_dir() / "conjugate_exceptions.txt"


def get_ignored_adverb_facets_filepath():
    return get_misc_dir() / "ignored_adverb_facets.txt"


def get_ignored_facet_prefixes_filepath():
    return get_misc_dir() / "ignored_facet_prefixes.txt"


def get_common_confusing_verbs_filepath():
    return get_misc_dir() / "common_confusing_verbs.txt"


def get_prepositions_filepath():
    return get_misc_dir() / "prepositions.txt"


def get_synonyms_tobe_filepath():
    return get_misc_dir() / "tobe_synonyms.txt"


def get_special_predicates_filepath():
    return get_misc_dir() / "special_predicates.txt"


def get_numeral_adjectives_filepath():
    return get_misc_dir() / "numeral_adjectives.txt"


def get_special_facet_connectors_filepath():
    return get_misc_dir() / "special_facet_connectors.txt"


def get_special_phrase_patterns_filepath():
    return get_misc_dir() / "special_phrase_patterns.json"


def get_redundant_predicate_prefixes_filepath():
    return get_misc_dir() / "redundant_predicate_prefixes.txt"


def get_redundant_predicate_suffixes_filepath():
    return get_misc_dir() / "redundant_predicate_suffixes.txt"


def get_facet_labels_filepath():
    return get_misc_dir() / "facet_labels.json"


def get_ignored_subgroups_filepath():
    return get_misc_dir() / "ignored_subgroups.txt"


def get_ignored_subparts_filepath():
    return get_misc_dir() / "ignored_subparts.txt"


def get_has_part_verbs_filepath():
    return get_misc_dir() / "has_part_verbs.txt"


def get_wn_wp_map_filepath():
    return UniversalFilePathHandler.RES_DIR / "wn_wp_map.csv"


def dir_path(path: str):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")
