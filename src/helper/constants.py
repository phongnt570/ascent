import json
from pathlib import Path
from typing import Set, Dict

from filepath_handler import get_modal_verbs_filepath, get_ignored_pronouns_in_obj_filepath, \
    get_ignored_one_word_objects_filepath, get_ignored_predicates_filepath, get_pronouns_filepath, \
    get_conjugate_exceptions_filepath, get_ignored_adverb_facets_filepath, get_ignored_facet_prefixes_filepath, \
    get_common_confusing_verbs_filepath, get_prepositions_filepath, get_synonyms_tobe_filepath, \
    get_special_predicates_filepath, get_numeral_adjectives_filepath, get_special_facet_connectors_filepath, \
    get_special_phrase_patterns_filepath, get_redundant_predicate_prefixes_filepath, \
    get_redundant_predicate_suffixes_filepath, get_ignored_subgroups_filepath, get_ignored_subparts_filepath, \
    get_has_part_verbs_filepath


def get_lines(path: Path) -> Set[str]:
    with path.open() as F:
        return set([line.strip().lower() for line in F if line.strip()])


def get_tsv_pairs(path: Path, lower=True) -> Dict[str, str]:
    mapping = {}
    with path.open() as F:
        for line in F:
            line = line.strip()
            if not line:
                continue
            ts = line.lower().split('\t')
            if not lower:
                ts = line.split('\t')
            mapping[ts[0]] = ts[1]
    return mapping


SPACY_MODEL_NAME = "en_core_web_md"

MODAL_VERBS = get_lines(get_modal_verbs_filepath())
PRONOUNS_IGNORED = get_lines(get_ignored_pronouns_in_obj_filepath())
ONE_WORD_OBJECT_IGNORED = get_lines(get_ignored_one_word_objects_filepath())
IGNORED_PREDICATES = get_lines(get_ignored_predicates_filepath())
ALL_PRONOUNS = get_lines(get_pronouns_filepath())
CONJUGATE_EXCEPTIONS = get_tsv_pairs(get_conjugate_exceptions_filepath())
IGNORED_ADVERB_FACETS = get_lines(get_ignored_adverb_facets_filepath())
IGNORED_FACET_PREFIXES = get_lines(get_ignored_facet_prefixes_filepath())
COMMON_CONFUSING_VERBS = get_lines(get_common_confusing_verbs_filepath())
PREPOSITIONS = get_lines(get_prepositions_filepath())
SYNONYMS_OF_TOBE = get_lines(get_synonyms_tobe_filepath())
SPECIAL_PREDICATES = get_lines(get_special_predicates_filepath())
NUMERAL_ADJECTIVES = get_lines(get_numeral_adjectives_filepath())
SPECIAL_FACET_CONNECTORS = get_lines(get_special_facet_connectors_filepath())
REDUNDANT_PREDICATE_PREFIXES = get_lines(get_redundant_predicate_prefixes_filepath())
REDUNDANT_PREDICATE_SUFFIXES = get_lines(get_redundant_predicate_suffixes_filepath())

with get_special_phrase_patterns_filepath().open() as f:
    SPECIAL_PHRASE_PATTERNS = json.load(f)

IGNORED_SUBGROUPS = get_lines(get_ignored_subgroups_filepath())
IGNORED_SUBPARTS = get_lines(get_ignored_subparts_filepath())
HAS_PART_VERBS = get_lines(get_has_part_verbs_filepath())
