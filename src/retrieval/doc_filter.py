"""Contains functions to compute the similarity scores of documents against the Wikipedia article of a given subject."""

from typing import List

from nltk.corpus.reader.wordnet import Synset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from filepath_handler import get_article_dir, get_wiki_path
from retrieval.grab_article import get_urls


def get_similarity_scores_of_all_articles(subject: Synset) -> List[float]:
    """Main function to compute the similarity scores for all articles regarding a given subject."""

    urls = get_urls(subject)
    article_list = [get_article(subject, i) for i in range(len(urls))]
    wikipedia_article = get_wikipedia_article(subject)

    if len(wikipedia_article.split()) <= 200:
        return [1.0] * len(article_list)

    similarity_scores = []
    for article in article_list:
        similarity_scores.append(compute_cosine_similarity(wikipedia_article, article))

    return similarity_scores


def compute_cosine_similarity(text1: str, text2: str) -> float:
    """Compute the pairwise similarity between two articles."""

    if text1 == text2:
        return 1.0

    try:
        return get_cosine_sim(text1, text2)[0][1]
    except:  # noqa
        return 0.0


def get_cosine_sim(*strings):
    """Compute all pairwise similarity scores between a list of articles."""

    vectors = [t for t in get_vectors(*strings)]
    return cosine_similarity(vectors)


def get_vectors(*strings):
    """Transform a list of articles into binary representation vectors.
    Stop words are discarded as default in Scikit-learn."""

    text = [t for t in strings]
    vectorizer = CountVectorizer(text)
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()


def get_article(subject: Synset, doc_id: int) -> str:
    """Return the article content given its id."""

    filepath = get_article_dir(subject) / "{}.txt".format(doc_id)
    if not filepath.exists():
        return ""
    with filepath.open() as f:
        return f.read()


def get_wikipedia_url(subject: Synset) -> str:
    with get_wiki_path(subject).open() as f:
        return f.readline().strip()


def get_wikipedia_article(subject: Synset) -> str:
    """Return the Wikipedia article of a subject."""

    wiki_url = get_wikipedia_url(subject)
    urls = get_urls(subject)

    for doc_id, url in enumerate(urls):
        if url.lower() == wiki_url.lower():
            return get_article(subject, doc_id)
