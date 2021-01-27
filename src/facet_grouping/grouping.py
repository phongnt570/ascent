import json
from collections import Counter
from typing import Counter as CounterType
from typing import Dict, Any, List, Tuple

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

from extraction.extractor import GENERAL_ASSERTION_KEY, SUBGROUP_ASSERTION_KEY, ASPECT_ASSERTION_KEY, SUBGROUPS_KEY, \
    STATISTICS_KEY
from facet_grouping.facet_clustering import facet_clustering, FacetCluster
from filepath_handler import get_final_kb_json_path, get_facet_labeled_json_path
from helper.constants import PREPOSITIONS
from retrieval.querying import has_hypernym
from triple_clustering.simple_assertion import SimpleAssertion, SimpleFacet


class AssertionCluster(object):
    def __init__(self, assertion_list: List[SimpleAssertion]):
        triple_counter: CounterType[SimpleAssertion] = Counter(assertion_list)
        representative = find_representative_triple(triple_counter)

        self.subj: str = representative.subj
        self.pred, self.obj = reorganize_utterances(representative.pred, representative.obj)
        self.count: int = len(assertion_list)

        self.expressions: List[CountedSimpleTriple] = [CountedSimpleTriple(assertion, count) for assertion, count in
                                                       triple_counter.most_common()]

        facet_counter: CounterType[SimpleFacet] = get_facet_counter(assertion_list)
        self.facets: List[FacetCluster] = sorted(facet_clustering(facet_counter), key=lambda facet: facet.count,
                                                 reverse=True)
        self.sources: List[str] = list(set(a.source for a in assertion_list if a.source is not None))

    def to_dict(self) -> dict:
        return {
            'subject': self.subj,
            'predicate': self.pred,
            'object': self.obj,
            'count': self.count,
            'expressions': [triple.to_dict() for triple in self.expressions],
            'facets': [facet.to_dict() for facet in self.facets],
            'sources': self.sources
        }


class CountedSimpleTriple(object):
    def __init__(self, assertion: SimpleAssertion, count: int):
        self.subj = assertion.subj
        self.pred, self.obj = reorganize_utterances(assertion.pred, assertion.obj)
        self.count = count

    def to_dict(self) -> dict:
        return {
            'subject': self.subj,
            'predicate': self.pred,
            'object': self.obj,
            'count': self.count,
        }


def find_representative_triple(triple_counter: Counter) -> SimpleAssertion:
    sorted_list = sorted(triple_counter.most_common(), key=lambda t: (-t[1], (len(t[0].pred) + len(t[0].obj))))

    if len(triple_counter) == 1:
        return sorted_list[0][0]

    most_counted = [assertion for assertion, count in sorted_list if count == sorted_list[0][1]]
    if len(most_counted) == 1:
        return most_counted[0]

    head_words = [assertion.get_obj_head_word() for assertion in most_counted]
    best_head_word = Counter(head_words).most_common(1)[0][0]

    for assertion in most_counted:
        if assertion.get_obj_head_word() == best_head_word:
            return assertion

    return most_counted[0]


def reorganize_utterances(pred: str, obj: str) -> Tuple[str, str]:
    # move all trailing prepositions from predicate to object
    tokens_of_predicate = pred.split()

    preps = []
    while len(tokens_of_predicate) > 0 and tokens_of_predicate[-1] in PREPOSITIONS:
        preps.append(tokens_of_predicate.pop())
    preps.reverse()
    preps.append(obj)

    pred = " ".join(tokens_of_predicate)
    obj = " ".join(preps)

    return pred, obj


def get_facet_counter(assertion_list: List[SimpleAssertion]) -> Counter:
    facet_list = [facet for assertion in assertion_list for facet in assertion.facets]
    facet_counter: CounterType[SimpleFacet] = Counter(facet_list)
    for facet in facet_counter.keys():
        label_counter: CounterType[str] = Counter([f.label for f in facet_list if f == facet])
        for label, _ in label_counter.most_common():
            if label is not None:
                facet.label = label

    return facet_counter


def group_subject_data(subject_data: Dict[str, Any]):
    clusters: List[List[SimpleAssertion]] = [[SimpleAssertion(assertion) for assertion in cluster] for cluster in
                                             subject_data["clusters"]]
    assertion_list = [AssertionCluster(cluster) for cluster in clusters]

    subject_data["clusters"] = [assertion.to_dict() for assertion in assertion_list if
                                len(assertion.pred) > 0]  # filter empty-predicate assertions

    return subject_data


def count_assertions(data):
    return sum([len(sub_data["clusters"]) for sub_data in data])


def count_facet(data):
    return sum([len(assertion["facets"]) for sub_data in data for assertion in sub_data["clusters"]])


def group_for_one_subject(subject: Synset):
    with get_facet_labeled_json_path(subject).open() as f:
        data = json.load(f)

    # remove incorrect subgroups
    to_be_removed = set()
    existed = set()
    for subgroup in data[SUBGROUPS_KEY]:
        name = subgroup["name"]
        s_ss = wn.synsets(name.replace(" ", "_"), "n")
        if len(s_ss) == 1:
            s_ss = s_ss[0]
            if (not has_hypernym(s_ss, subject)) or s_ss.name() in existed:
                to_be_removed.add(name)
            else:
                subgroup["ssid"] = s_ss.name()
                existed.add(s_ss.name())
    data[SUBGROUPS_KEY] = [sg for sg in data[SUBGROUPS_KEY] if sg["name"] not in to_be_removed]
    data[SUBGROUP_ASSERTION_KEY] = [a for a in data[SUBGROUP_ASSERTION_KEY] if a["subject"] not in to_be_removed]

    # remove duplicated subgroups
    ids_to_be_removed = set()
    names_existed = set()
    for i, subgroup in enumerate(data[SUBGROUPS_KEY]):
        name = subgroup["name"]
        if name in names_existed:
            ids_to_be_removed.add(i)
        else:
            names_existed.add(name)
    data[SUBGROUPS_KEY] = [sg for i, sg in enumerate(data[SUBGROUPS_KEY]) if i not in ids_to_be_removed]
    data[STATISTICS_KEY]["num_subgroups"] = len(data[SUBGROUPS_KEY])

    # find representative for each assertion cluster
    for subject_data in (data[GENERAL_ASSERTION_KEY] + data[SUBGROUP_ASSERTION_KEY] + data[ASPECT_ASSERTION_KEY]):
        group_subject_data(subject_data)

    # remove empty subgroup/subpart
    data[SUBGROUP_ASSERTION_KEY] = sorted([s for s in data[SUBGROUP_ASSERTION_KEY] if len(s["clusters"]) > 0],
                                          key=lambda x: -len(x["clusters"]))
    data[ASPECT_ASSERTION_KEY] = sorted([s for s in data[ASPECT_ASSERTION_KEY] if len(s["clusters"]) > 0],
                                        key=lambda x: -len(x["clusters"]))

    # update statistics
    data[STATISTICS_KEY].update({
        "num_subgroups": len(data[SUBGROUP_ASSERTION_KEY]),
        "num_aspects": len(data[ASPECT_ASSERTION_KEY]),
        "num_canonical_facets": sum([len(a["facets"])
                                     for subject_data in (
                                             data[GENERAL_ASSERTION_KEY] +
                                             data[SUBGROUP_ASSERTION_KEY] +
                                             data[ASPECT_ASSERTION_KEY]
                                     ) for a in subject_data["clusters"]]),
        "num_canonical_general_assertions": sum([len(name["clusters"]) for name in data[GENERAL_ASSERTION_KEY]]),
        "num_canonical_subgroup_assertions": sum([len(name["clusters"]) for name in data[SUBGROUP_ASSERTION_KEY]]),
        "num_canonical_aspect_assertions": sum([len(name["clusters"]) for name in data[ASPECT_ASSERTION_KEY]]),
    }),

    with get_final_kb_json_path(subject).open("w+", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=False)

    return data
