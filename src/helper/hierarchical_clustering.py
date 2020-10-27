from typing import List, TypeVar

import numpy as np
from sklearn.cluster import AgglomerativeClustering

T = TypeVar('T')


def hierarchical_clustering(object_list: List[T], distance_matrix: np.ndarray, linkage: str,
                            distance_threshold: float) -> List[List[T]]:
    """The clustering function to cluster any list given the distance matrix between its members."""

    if len(object_list) <= 1:
        return [object_list]

    clustering_factory = AgglomerativeClustering(
        n_clusters=None,
        affinity="precomputed",
        distance_threshold=distance_threshold,
        compute_full_tree=True,
        linkage=linkage
    )

    clustering_factory.fit(distance_matrix)
    labels = clustering_factory.labels_

    object_array = np.array(object_list)
    clusters = []
    for cluster_label in range(np.max(labels) + 1):
        cluster = object_array[labels == cluster_label]
        clusters.append(cluster)

    return clusters
