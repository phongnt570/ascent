"""Contains functions to search for relevant documents using the Bing API."""

import http.client
import json
import logging
from typing import List, Tuple, Set
from urllib.parse import quote_plus

from nltk.corpus.reader.wordnet import Synset

from helper.constants import MANUAL_WN2WP, EN_WIKIPEDIA_PREFIX
from retrieval.querying import get_search_query, get_wikipedia_search_query

SEARCH_BATCH_SIZE = 50  # maximum number of urls can be retrieved per request to the Bing API

MAX_SEARCH_STEP = 20

logger = logging.getLogger(__name__)


def search_for_subject(subject: Synset, num_urls: int, subscription_key: str, custom_config: str,
                       host: str, path: str) -> Tuple[List[Tuple[str, str, str]], str, str]:
    """Perform the search phase for one particular subject."""

    query = get_search_query(subject)

    logger.info(f"Subject {subject.name()} - Search query: `{query}`")

    urls: Set[str] = set()
    results: List[Tuple[str, str, str]] = []
    wiki_links: List[str] = []

    offset = 0
    step = 0
    while len(urls) < num_urls:
        search_result_json = bing_search(search_query=query,
                                         count=SEARCH_BATCH_SIZE,
                                         offset=offset,
                                         subscription_key=subscription_key,
                                         custom_config=custom_config,
                                         host=host,
                                         path=path)

        try:
            for url, title, snippet in parse_content_from_search_result(search_result_json):
                if url not in urls:
                    urls.add(url)
                    results.append((url, title, snippet))
                    if url.startswith(EN_WIKIPEDIA_PREFIX):
                        wiki_links.append(url)
                    if len(urls) >= num_urls:
                        break
        except Exception:
            break

        offset += SEARCH_BATCH_SIZE

        step += 1
        if step >= MAX_SEARCH_STEP:
            break
    if subject.name() in MANUAL_WN2WP:
        logger.info("Detected manual WordNet-Wikipedia linking")
        wiki = EN_WIKIPEDIA_PREFIX + quote_plus(MANUAL_WN2WP[subject.name()]["wikipedia"])
        wiki_map_source = MANUAL_WN2WP[subject.name()]["source"]
    else:
        if len(wiki_links) == 0:
            wiki_links = search_wiki(subject, subscription_key, custom_config, host, path)
        wiki = wiki_links[0]
        for w in wiki_links:
            if "List_" in w:
                continue
            if "(disambiguation)" in w:
                continue
            if "Category:" in w:
                continue
            if "Template:" in w:
                continue
            wiki = w
            break
        wiki_map_source = "BING"

    # Add Wikipedia article
    if wiki not in urls:
        results[-1] = (wiki, "{} - Wikipedia".format(wiki[(wiki.rindex("/") + 1):]), "")

    return results, wiki, wiki_map_source


def bing_search(search_query: str, count: int, offset: int, subscription_key: str, custom_config: str, host: str,
                path: str) -> str:
    """Performs a Bing Web search and returns the results."""

    headers = {'Ocp-Apim-Subscription-Key': subscription_key}
    conn = http.client.HTTPSConnection(host)
    query = quote_plus(search_query)
    conn.request("GET",
                 path + "?q=" + query + "&customconfig="
                 + custom_config + "&mkt=en-US&safesearch=Moderate"
                 + "&count=" + str(count) + "&offset=" + str(offset),
                 headers=headers)
    response = conn.getresponse()

    # logger.info([k + ": " + v for (k, v) in response.getheaders()
    #             if k.startswith("BingAPIs-") or k.startswith("X-MSEdge-")])

    return response.read().decode("utf8")


def parse_content_from_search_result(search_result_json: str) -> List[Tuple[str, str, str]]:
    """Parse the json result returned by the Bing API to get list of URLs."""

    json_obj = json.loads(search_result_json)

    web_pages = json_obj["webPages"]
    values = web_pages["value"]

    results = []
    for web in values:
        if web["language"] == "en":
            results.append((web["url"], web["name"], web["snippet"]))

    return results


def search_wiki(subject: Synset, subscription_key: str, custom_config: str, host: str, path: str) \
        -> List[str]:
    query = get_wikipedia_search_query(subject)
    search_result_json = bing_search(search_query=query,
                                     count=10,
                                     offset=0,
                                     subscription_key=subscription_key,
                                     custom_config=custom_config,
                                     host=host,
                                     path=path)

    results = []
    for url, title, snippet in parse_content_from_search_result(search_result_json):
        results.append(url)
    return results
