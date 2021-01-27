"""Contains functions to extract subgroups and associated terms of a given subject."""

from collections import Counter
from typing import List, Set, Dict

import numpy as np
from nltk.corpus import wordnet
from sklearn.cluster import AgglomerativeClustering
from spacy import symbols
from spacy.language import Language
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from spacy.tokens.token import Token

from extraction.assertion import Assertion, simplify_predicate
from extraction.supporting import remove_redundancy_from_subgroup_chunk, lemmatize_token_list, get_target, \
    find_short_phrase, find_compound_noun, find_long_phrase, chunk_ends_with_tokens, is_noun_or_proper_noun
from helper.constants import IGNORED_SUBGROUPS, HAS_PART_VERBS, IGNORED_SUBPARTS

SALIENT_THRESHOLD = 3

SUBGROUP_CLUSTERING_THRESHOLD = 0.2

FIRST_SECOND_POSSESSIVE_PRONOUNS = {"your", "our", "my"}  # to be discarded


# Subgroup extraction

class Subgroup(object):
    """Class representing a subgroup. Construct it by giving a cluster of noun chunks.
    Most occurring chunk will be chosen to be the representative."""

    def __init__(self, chunk_list: List[List[Token]]):
        self.phrase_counter = Counter([lemmatize_token_list(chunk) for chunk in chunk_list])
        self.name = self.phrase_counter.most_common(1)[0][0]

    def get_frequency(self) -> int:
        return sum(self.phrase_counter.values())

    def merge(self, other) -> None:
        self.phrase_counter = self.phrase_counter + other.phrase_counter
        if len(self.name) > len(other.name):
            self.name = other.name

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "count": self.get_frequency(),
            "occurrences": [
                {
                    "phrase": phrase,
                    "count": count
                }
                for phrase, count in self.phrase_counter.most_common()
            ]
        }


def extract_subgroups(doc_list: List[Doc], subject: str, spacy_nlp: Language) -> List[Subgroup]:
    """Main function to extract subgroups. Called by `extractor.py`."""

    # split subject into tokens, e.g. "polar bear" -> ["polar", "bear"]
    tokens_of_subject = subject.lower().split()

    # extract candidate noun chunks
    subgroup_chunks = []
    for doc in doc_list:
        subgroup_chunks.extend(single_subgroup_extraction_run(doc, tokens_of_subject))

    # filter
    to_be_removed = set()
    for ind, subgroup_chunk in enumerate(subgroup_chunks):
        if subgroup_chunk[0].lemma_.lower() in IGNORED_SUBGROUPS:
            to_be_removed.add(ind)
    subgroup_chunks = [chunk for idn, chunk in enumerate(subgroup_chunks) if idn not in to_be_removed]

    # cluster the noun chunks
    subgroup_list = subgroup_clustering(subgroup_chunks, tokens_of_subject, spacy_nlp)

    # merge groups, e.g. `male canadian lynx` merged with `canadian lynx`
    subgroup_list = merge_subgroups(subgroup_list)

    # remove infrequent subgroups and sort by decreasing frequency
    return sorted([subgroup for subgroup in subgroup_list if subgroup.get_frequency() >= SALIENT_THRESHOLD],
                  key=lambda subgroup: -subgroup.get_frequency())


def single_subgroup_extraction_run(doc: Doc, tokens_of_subject: List[str]) -> List[List[Token]]:
    """Subgroups are chunks ending with the subject.
    For example, given the subject `lynx`, subgroups can be `Canadian Lynx`, `mother lynx`, etc.
    Each chunk is a list of Tokens, after removing unnecessary words."""

    subgroup_chunk_list = []
    for chunk in doc.noun_chunks:
        if chunk.root == chunk[-1] and chunk_ends_with_tokens(chunk, tokens_of_subject):
            # discard conjunctive phrases, e.g, "both Asian and African elephants"
            if any([(token.dep == symbols.cc or token.dep == symbols.appos) for token in chunk]):
                continue

            simplified_chunk = remove_redundancy_from_subgroup_chunk(chunk)

            # discard entities, e.g., Kristen Steward, Stanford University
            if all(token.ent_type_ for token in simplified_chunk):
                continue

            # double-check the ending requirement
            if lemmatize_token_list(simplified_chunk).endswith(" ".join(tokens_of_subject)):
                # noun chunks should not be too long
                if len(tokens_of_subject) < len(simplified_chunk) <= len(tokens_of_subject) + 3:
                    if all([token.is_alpha for token in simplified_chunk]):  # only alphabetic terms
                        subgroup_chunk_list.append(simplified_chunk)

    return subgroup_chunk_list


def subgroup_clustering(subgroup_chunk_list: List[List[Token]], tokens_of_subject: List[str], spacy_nlp: Language) \
        -> List[Subgroup]:
    """Group same noun chunks using word2vec and Hierarchical Clustering."""

    if len(subgroup_chunk_list) == 0:
        return []

    if len(subgroup_chunk_list) == 1:
        return [Subgroup(subgroup_chunk_list)]

    clustering_factory = AgglomerativeClustering(
        n_clusters=None,
        affinity="precomputed",
        distance_threshold=SUBGROUP_CLUSTERING_THRESHOLD,
        compute_full_tree=True,
        linkage="complete"
    )

    distance_matrix = compute_distance_matrix(subgroup_chunk_list, tokens_of_subject, spacy_nlp)
    clustering_factory.fit(distance_matrix)
    labels = clustering_factory.labels_

    np_subgroup_array = np.array(subgroup_chunk_list, dtype="object")
    subgroup_list = []
    for c in range(np.max(labels) + 1):
        cluster = list(np_subgroup_array[labels == c])
        subgroup_list.append(Subgroup(cluster))

    return subgroup_list


def compute_distance_matrix(subgroup_chunk_list: List[List[Token]], tokens_of_subject: List[str],
                            spacy_nlp: Language) -> np.ndarray:
    """Distance matrix used for clustering the subgroup noun chunks. Essentially based on word2vec vector similarity.
    Also take care of antonyms retrieved by wordnet, because word2vec is usually confused with these cases
    (e.g. `male` and `female` have high similarity score due to word2vec)"""

    vectors = np.array(
        [get_normalized_vector_of_chunk(chunk[:-len(tokens_of_subject)], spacy_nlp) for chunk in subgroup_chunk_list])
    similarity_matrix = np.matmul(vectors, vectors.T)

    # take care of antonyms
    antonyms_list = [get_antonyms_of_token_list(chunk[:-len(tokens_of_subject)]) for chunk in subgroup_chunk_list]
    words_list = [set([token.lemma_.lower() for token in chunk[:-len(tokens_of_subject)]]) for chunk in
                  subgroup_chunk_list]

    for i, cur_antonyms in enumerate(antonyms_list):
        cur_words = words_list[i]
        for j in range(i + 1, len(antonyms_list)):
            nex_antonyms = antonyms_list[j]
            nex_words = words_list[j]
            if (cur_antonyms & nex_words) or (cur_words & nex_antonyms):  # intersections
                similarity_matrix[i][j] = 0.0
                similarity_matrix[j][i] = 0.0

    return 1 - similarity_matrix


def get_normalized_vector_of_chunk(token_list: List[Token], spacy_nlp: Language) -> np.ndarray:
    """Average the vectors of all tokens, except for the ones with missing vector.
    Normalize the resulting vector at the end by dividing it by its norm-2."""

    vector = np.zeros(spacy_nlp.vocab.vectors_length)
    num_tokens = 0
    for token in token_list:
        if spacy_nlp.vocab.has_vector(token.lemma_.lower()):
            vector += spacy_nlp.vocab.get_vector(token.lemma_.lower())
        num_tokens += 1

    if num_tokens == 0:
        return vector

    vector = vector / num_tokens
    norm = np.linalg.norm(vector)

    return vector if norm == 0 else vector / norm


def get_antonyms_of_token_list(token_list: List[Token]) -> Set[str]:
    """Using wordnet to retrieve all antonyms of every token in the given list."""

    antonyms = set()
    for token in token_list:
        word = token.lemma_.lower()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.antonyms():
                    antonyms.add(lemma.antonyms()[0].name())

    return antonyms


def merge_subgroups(subgroup_list: List[Subgroup]) -> List[Subgroup]:
    """Post-processing the clusters by merging long names with short sub-name.
    For example, `male canadian lynx` is merged with `canadian lynx`"""

    if len(subgroup_list) <= 1:
        return subgroup_list

    # sort the subgroup with increasing name length
    subgroup_list = sorted(list(subgroup_list), key=lambda subgroup: len(subgroup.name))

    # merge
    has_been_merged = set()
    for i in range(len(subgroup_list)):
        if i in has_been_merged:
            continue
        short_name = subgroup_list[i].name
        for j in range(i + 1, len(subgroup_list)):
            if j in has_been_merged:
                continue
            long_name = subgroup_list[j].name
            if long_name.endswith(" " + short_name) or long_name == short_name:
                subgroup_list[i].merge(subgroup_list[j])
                has_been_merged.add(j)

    return [subgroup for i, subgroup in enumerate(subgroup_list) if i not in has_been_merged]


# Subpart extraction

class Subpart(object):
    """Class representing an associated term."""

    def __init__(self, name: str):
        self.name = name
        self.phrase_counter = Counter()
        self.chunk_list = []

    def add_phrase(self, phrase: str, count: int = 1) -> None:
        self.phrase_counter += {phrase: count}

    def get_frequency(self) -> int:
        return sum(self.phrase_counter.values())

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "count": self.get_frequency(),
            "occurrences": [
                {
                    "phrase": phrase,
                    "count": count
                }
                for phrase, count in self.phrase_counter.most_common()
            ]
        }

    def add_chunk(self, chunk: Span) -> None:
        self.chunk_list.append(chunk)


def extract_subparts(doc_list: List[Doc], assertion_list: List[Assertion], subject: str) -> List[Subpart]:
    """Main function to extract associated terms of a subject. Called by `extractor.py`."""

    # tokens of the subject string, used to retrieve relevant co-reference
    tokens_of_subject = subject.lower().split()

    # head noun of subject, used to retrieve possible relevant co-reference
    subject_head_noun = tokens_of_subject[-1]

    seen_nouns = set()
    string_2_subpart = {}

    # CASE 1: assertion pattern <subject/subgroup; have; subpart>
    for assertion in assertion_list:
        if chunk_ends_with_tokens(assertion.full_subj, tokens_of_subject) or \
                any([chunk_ends_with_tokens(cluster.main, tokens_of_subject)
                     for cluster in get_coref_clusters_of_token(assertion.subj)]):
            if (not assertion.is_synthetic) and \
                    simplify_predicate(assertion.full_pred) in HAS_PART_VERBS and \
                    assertion.verb.pos != symbols.AUX and \
                    assertion.full_pred[-1] == assertion.verb and \
                    assertion.obj is not None and \
                    is_noun_or_proper_noun(assertion.obj):

                # pre-check
                if not pre_check_subpart(assertion.obj, subject_head_noun):
                    continue

                # ignore negation
                if any([token.dep == symbols.neg for token in assertion.verb.children]):
                    continue

                add_subpart_to_set(string_2_subpart, assertion.obj,
                                   phrase="{} {} {}".format(assertion.full_subj,
                                                            " ".join([str(token) for token in assertion.full_pred]),
                                                            assertion.full_obj).lower())
                seen_nouns.add(assertion.obj)

    # CASE 2: noun_chunk = `possessive` + `subpart`, where `possessive` refers to subject/subgroup
    for doc in doc_list:
        for noun_chunk in doc.noun_chunks:
            # pre-check
            if not pre_check_subpart(noun_chunk.root, subject_head_noun):
                continue

            # already seen in CASE 1
            if noun_chunk.root in seen_nouns:
                continue

            # possessive pronoun, e.g. "their tail"
            if noun_chunk[0].dep == symbols.poss and \
                    noun_chunk[0].lower_ not in FIRST_SECOND_POSSESSIVE_PRONOUNS and \
                    noun_chunk[0].head == noun_chunk.root:
                for cluster in get_coref_clusters_of_token(noun_chunk[0]):
                    if chunk_ends_with_tokens(cluster.main, tokens_of_subject):
                        add_subpart_to_set(string_2_subpart, noun_chunk.root)
                        break
            # possessive noun, e.g. "lynx's tail"
            else:
                possessive_noun = get_target(noun_chunk.root, symbols.poss)
                if possessive_noun is not None and \
                        chunk_ends_with_tokens(find_short_phrase(possessive_noun), tokens_of_subject):
                    add_subpart_to_set(string_2_subpart, noun_chunk.root)

    # remove infrequent subparts and sort by decreasing frequency
    return sorted([subpart for subpart in string_2_subpart.values() if subpart.get_frequency() >= SALIENT_THRESHOLD],
                  key=lambda subpart: -subpart.get_frequency())


def pre_check_subpart(head_noun: Token, subject_head_noun: str) -> bool:
    """Set of rules to filter out invalid terms."""

    # ignored subparts
    if head_noun.lemma_.lower() in IGNORED_SUBPARTS:
        return False

    # contains subject
    if head_noun.lemma_.lower() == subject_head_noun:
        return False

    # non-alphabetic terms
    if not head_noun.is_alpha:
        return False

    # pronouns
    if head_noun.pos == symbols.PRON:
        return False

    return True


def add_subpart_to_set(string_2_subpart: Dict[str, Subpart], head_noun: Token, phrase: str = None) -> None:
    """Extract subpart given the head noun and add it to the subpart set."""

    subpart_string = find_compound_noun(head_noun)  # e.g., "tuft" -> "ear tuft" if possible
    if subpart_string == '-pron-':  # NOTE: SpaCy's lemma form of pronouns
        return

    if subpart_string not in string_2_subpart:
        string_2_subpart[subpart_string] = Subpart(subpart_string)

    if phrase is None:
        chunk = find_long_phrase(head_noun, prep_in={"of", "on", "in", "at"})
        string_2_subpart[subpart_string].add_chunk(chunk)

        phrase = chunk.lower_

    string_2_subpart[subpart_string].add_phrase(phrase)


def get_coref_clusters_of_token(token: Token) -> List:
    """Get the co-reference clusters of a token."""

    # noinspection PyProtectedMember
    return token._.coref_clusters
