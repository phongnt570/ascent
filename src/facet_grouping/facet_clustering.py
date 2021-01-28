from typing import Counter as CounterType, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from helper.hierarchical_clustering import hierarchical_clustering
from triple_clustering.simple_assertion import SimpleFacet


class FacetCluster(object):
    def __init__(self, facet_list: List[SimpleFacet], facet_counter: CounterType[SimpleFacet]):
        representative = find_representative_facet(facet_list, facet_counter)

        self.value: str = representative.get_facet_str().lower()
        if self.value.startswith("both "):
            tmp = self.value[len("both "):]
            if tmp:
                self.value = tmp

        self.label: str = representative.label
        self.count: int = sum([facet_counter[facet] for facet in facet_list])

        self.expressions: List[CountedSimpleFacet] = [CountedSimpleFacet(facet, facet_counter[facet]) for facet in
                                                      facet_list]
        self.expressions = sorted(self.expressions, key=lambda facet: facet.count, reverse=True)

    def to_dict(self) -> dict:
        return {
            'value': self.value,
            'label': self.label,
            'count': self.count,
            'expressions': [facet.to_dict() for facet in self.expressions]
        }


class CountedSimpleFacet(object):
    def __init__(self, facet: SimpleFacet, count: int):
        self.value: str = facet.get_facet_str().lower()
        self.count: int = count

    def to_dict(self) -> dict:
        return {
            'value': self.value,
            'count': self.count
        }


def find_representative_facet(facet_list: List[SimpleFacet], facet_counter: CounterType[SimpleFacet]) -> SimpleFacet:
    facet_list = sorted(facet_list, key=lambda facet: facet_counter[facet], reverse=True)

    top_facets = [facet for facet in facet_list if facet_counter[facet] == facet_counter[facet_list[0]]]
    top_facets = sorted(top_facets, key=lambda facet: len(facet.get_facet_str()))

    return top_facets[0]


def facet_clustering(facet_counter: CounterType[SimpleFacet]) -> List[FacetCluster]:
    facet_list: List[SimpleFacet] = list(facet_counter.keys())

    if len(facet_list) <= 1:
        cluster_list = [facet_list]
    else:
        cluster_list = hierarchical_clustering(
            facet_list,
            distance_matrix=compute_facet_distance_matrix(facet_list),
            linkage="single",
            distance_threshold=0.3
        )
    return [FacetCluster(cluster, facet_counter) for cluster in cluster_list if len(cluster) > 0]


def compute_facet_distance_matrix(facet_list: List[SimpleFacet]) -> np.ndarray:
    distance_matrix = compute_facet_cosine_distance_matrix(facet_list)

    for row_ind, facet1 in enumerate(facet_list):
        distance_matrix[row_ind][row_ind] = 0.0
        for col_ind in range(row_ind + 1, len(facet_list)):
            facet2 = facet_list[col_ind]
            if facet1.label != facet2.label:
                distance_matrix[row_ind][col_ind] = 1.0
                distance_matrix[col_ind][row_ind] = 1.0
            else:
                if have_same_head_word(facet1, facet2):
                    distance_matrix[row_ind][col_ind] = 0.0
                    distance_matrix[col_ind][row_ind] = 0.0
                else:
                    tokens_1 = facet1.get_facet_str().split()
                    tokens_2 = facet2.get_facet_str().split()

                    if min([len(tokens_1), len(tokens_2)]) >= 3:
                        if facet1.get_facet_str() in facet2.get_facet_str() \
                                or facet2.get_facet_str() in facet1.get_facet_str():
                            distance_matrix[row_ind][col_ind] = 0.0
                            distance_matrix[col_ind][row_ind] = 0.0
                            continue

                    if len(tokens_1) > 1 or len(tokens_2) > 1:
                        distance_matrix[row_ind][col_ind] = 1.0
                        distance_matrix[col_ind][row_ind] = 1.0

    return distance_matrix


def compute_facet_cosine_distance_matrix(facet_list: List[SimpleFacet]) -> np.ndarray:
    matrix = np.array([facet.get_vector() for facet in facet_list])

    similarity = cosine_similarity(matrix)
    similarity = np.nan_to_num(similarity)

    return 1 - similarity


def have_same_head_word(facet1: SimpleFacet, facet2: SimpleFacet) -> bool:
    head_word_1 = facet1.get_head_word()
    head_word_2 = facet2.get_head_word()

    if head_word_1.lemma_.lower() == head_word_2.lemma_.lower():
        return True

    if head_word_1.lower_ == head_word_2.lower_ + "s" or head_word_2.lower_ == head_word_1.lower_ + "s":
        return True

    return False
