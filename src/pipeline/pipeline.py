from configparser import ConfigParser
from typing import List, Optional

from nltk.corpus import wordnet as wn

from pipeline.article_filtering_module import ArticleFilteringModule
from pipeline.article_grab_module import ArticleGrabModule
from pipeline.bing_search_module import BingSearchModule
from pipeline.extraction_module import ExtractionModule
from pipeline.facet_grouping_module import FacetGroupingModule
from pipeline.facet_labeling_module import FacetLabelingModule
from pipeline.module_interface import Module
from pipeline.triple_clustering_module import TripleClusteringModule
from static_resource import StaticResource

DEFAULT_MODULES = [
    "bing_search",
    "article_grab",
    "article_filtering",
    "extraction",
    "triple_clustering",
    "facet_labeling",
    "facet_grouping",
]


class Pipeline(object):
    def __init__(self, config: ConfigParser):
        self._config = config

        self._modules = [get_module_by_name(name, config) for name in DEFAULT_MODULES]

    def run(self, subject_list: List[str], from_module: int, to_module: int, **kwargs):
        # load SpaCy, once for all
        for i in range(from_module, to_module + 1):
            if DEFAULT_MODULES[i] in {"extraction", "triple_clustering", "facet_grouping"}:
                StaticResource.nlp()
                break

        # HACK: overcome nltk.corpus multi-processing error
        [wn.synset(subject) for subject in subject_list]

        # execute modules
        for module in self._modules[from_module:(to_module + 1)]:
            module.run(subject_list, **kwargs)

    def print_modules(self):
        for ind, module in enumerate(self._modules):
            print(f"[{ind}] {module}")

    def __len__(self):
        return len(self._modules)


def get_module_by_name(module_name: str, config: ConfigParser) -> Optional[Module]:
    if module_name == "bing_search":
        return BingSearchModule(config=config)

    elif module_name == "article_grab":
        return ArticleGrabModule(config=config)

    elif module_name == "article_filtering":
        return ArticleFilteringModule(config=config)

    elif module_name == "extraction":
        return ExtractionModule(config=config)

    elif module_name == "triple_clustering":
        return TripleClusteringModule(config=config)

    elif module_name == "facet_labeling":
        return FacetLabelingModule(config=config)

    elif module_name == "facet_grouping":
        return FacetGroupingModule(config=config)

    return None
