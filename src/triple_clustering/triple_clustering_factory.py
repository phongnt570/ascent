"""The BERT-based triple clustering mechanism."""

from typing import List, Dict, Tuple

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from transformers import RobertaTokenizer, RobertaForSequenceClassification

from triple_clustering.simple_assertion import SimpleAssertion


class TripleClusteringFactory(object):
    """Class for BERT-based clustering factory."""

    def __init__(self, model_path: str, device, distance_threshold: float, batch_size: int,
                 top_n: int = 5):
        super().__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaForSequenceClassification.from_pretrained(model_path)
        self.device = device
        self.model.eval()

        if self.device != "cpu":
            self.model.to(self.device)

        self.distance_threshold = distance_threshold
        self.batch_size = batch_size
        self.top_n = top_n

    def cluster(self, assertion_list: List[SimpleAssertion]) -> List[List[SimpleAssertion]]:
        """Group assertions of identical meaning into same groups."""

        # grouping assertions having same lower-cased form
        triple2duplicate_list = self.group_same_triples(assertion_list)

        if len(triple2duplicate_list) == 1:
            return [assertion_list]

        # list of distinct triples
        distinct_triple_list = [triple for triple, _ in sorted(triple2duplicate_list.items(),
                                                               key=lambda x: (
                                                                   -len(x[1]),
                                                                   len(x[0].get_triple_str()),
                                                                   x[0].pred,
                                                                   x[0].obj)
                                                               )
                                ]

        # indexes of triple pairs to be processed
        to_be_processed_index_list = self.get_list_of_triple_pairs_to_be_processed(distinct_triple_list)

        # convert assertions to BERT-like sentences
        to_be_processed_sentence_list = [
            (self.prepare(distinct_triple_list[idx[0]]), self.prepare(distinct_triple_list[idx[1]])) for idx in
            to_be_processed_index_list]

        # compute BERT scores
        bert_scores = self.compute_bert_based_dissimilarity(to_be_processed_sentence_list)

        # building distance matrix
        distance_matrix = np.ones((len(distinct_triple_list), len(distinct_triple_list)))
        for idx, pair in enumerate(to_be_processed_index_list):
            score = bert_scores[idx]
            i, j = pair
            distance_matrix[i][j] = score
            distance_matrix[j][i] = score

        # special cases
        for i in range(distance_matrix.shape[0]):
            distance_matrix[i][i] = 0.0
            object_i = distinct_triple_list[i].get_simplified_object()
            for j in range(i + 1, distance_matrix.shape[1]):
                if distinct_triple_list[j].pred != distinct_triple_list[i].pred:
                    continue
                object_j = distinct_triple_list[j].get_simplified_object()
                if object_i == object_j:
                    distance_matrix[i][j] = 0.0
                    distance_matrix[j][i] = 0.0

        # hierarchical clustering
        clustering = AgglomerativeClustering(n_clusters=None, affinity="precomputed",
                                             distance_threshold=self.distance_threshold,
                                             compute_full_tree=True, linkage="single")
        clustering.fit(distance_matrix)
        labels = clustering.labels_
        np_distinct_triple_array = np.array(distinct_triple_list)
        clusters = []
        for label in range(np.max(labels) + 1):
            cluster = [a for a in np_distinct_triple_array[labels == label]]
            alias = []
            for a in cluster:
                alias.extend(triple2duplicate_list[a])
            cluster.extend(alias)
            clusters.append(cluster)

        return clusters

    @staticmethod
    def prepare(assertion: SimpleAssertion) -> str:
        """Prepare BERT-based input sentence."""

        return ' '.join(['[subj]', assertion.pred, '[u-sep]', assertion.obj])

    @staticmethod
    def group_same_triples(assertion_list: List[SimpleAssertion]) -> Dict[SimpleAssertion, List[SimpleAssertion]]:
        """Triples having same utterances are grouped together."""

        triple2duplicate_list = {}
        for assertion in assertion_list:
            if assertion in triple2duplicate_list:
                triple2duplicate_list[assertion].append(assertion)
            else:
                triple2duplicate_list[assertion] = []

        return triple2duplicate_list

    def compute_bert_based_dissimilarity(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Distance between two triples is the p_0 probability obtained by the BERT-based classification model."""

        forward_probs = []
        with torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i:(i + self.batch_size)]
                input_batch = self.tokenizer.batch_encode_plus(
                    batch,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=32
                )

                if self.device != "cpu":
                    for k in input_batch:
                        input_batch[k] = input_batch[k].to(self.device)

                logits = self.model(**input_batch)[0]
                batch_probs = torch.softmax(logits, dim=1).tolist()

                # probabilities of being not paraphrase = distance between 2 triples
                batch_probs = [p[0] for p in batch_probs]
                forward_probs.extend(batch_probs)

        return forward_probs

    def get_list_of_triple_pairs_to_be_processed(self, triple_list: List[SimpleAssertion]) -> List[Tuple[int, int]]:
        """Tricks to lower the search space for each triple, using pure word2vec."""

        # vector for (predicate + object)
        similarity_matrix = compute_word2vec_similarity_matrix(triple_list, includes_predicate=True)

        # vector for object only, stop words and punctuations discarded
        object_similarity_matrix = compute_word2vec_similarity_matrix(triple_list, includes_predicate=False)

        # sort by decreasing similarity
        ind = np.argsort(-similarity_matrix, axis=1)
        obj_ind = np.argsort(-object_similarity_matrix, axis=1)

        # extract top similar triples for each row
        pairs = []
        for i in range(ind.shape[0]):
            local_ind_pairs = set()
            local_obj_ind_pairs = set()

            for j in range(ind.shape[1]):
                # top pairs using (predicate + object) vector
                if ind[i][j] > i and len(local_ind_pairs) < self.top_n:
                    local_ind_pairs.add((i, ind[i][j]))
                # top pairs using only object vector
                if obj_ind[i][j] > i and len(local_obj_ind_pairs) < self.top_n:
                    local_obj_ind_pairs.add((i, obj_ind[i][j]))

                if len(local_ind_pairs) >= self.top_n and len(local_obj_ind_pairs) >= self.top_n:
                    break

            same_head_word_pairs = set()

            head_word_i = triple_list[i].get_obj_head_word()

            for j in range(i + 1, ind.shape[0]):
                # same head words
                head_word_j = triple_list[j].get_obj_head_word()
                if head_word_i == head_word_j and len(same_head_word_pairs) < self.top_n:
                    same_head_word_pairs.add((i, j))
                if len(same_head_word_pairs) >= self.top_n:
                    break

            # unite pair sets
            pairs.extend(
                local_ind_pairs.union(local_obj_ind_pairs).union(same_head_word_pairs))

        # filter antonyms
        to_be_removed = set()
        for ind, (i, j) in enumerate(pairs):
            obj1 = triple_list[i].obj
            obj2 = triple_list[j].obj
            shorter = obj1 if len(obj1) < len(obj2) else obj2
            longer = obj1 if len(obj1) > len(obj2) else obj2
            if longer in {"in" + shorter, "un" + shorter, "ir" + shorter}:
                to_be_removed.add(ind)
        pairs = [pair for ind, pair in enumerate(pairs) if ind not in to_be_removed]

        return pairs


def compute_word2vec_similarity_matrix(triple_list, includes_predicate=True) -> np.ndarray:
    """Fast computation for vector-vector cosine similarity."""
    if not includes_predicate:
        matrix = np.array([triple.get_object_vector() for triple in triple_list])
    else:
        matrix = np.array([triple.get_vector() for triple in triple_list])

    return cosine_similarity(matrix)
