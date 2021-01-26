"""Contains functions to scrap the text from URLs given by Bing."""

import logging
import re
from typing import Tuple, List, Union

from newspaper import Article
from nltk.corpus.reader.wordnet import Synset

from filepath_handler import get_article_dir, get_url_path

logger = logging.getLogger(__name__)


def grab_text(subject: Synset, id_url: Tuple[int, str]) -> Union[str, None]:
    """Main function to crawl text."""

    doc_id, url = id_url
    filepath = get_article_dir(subject) / f"{doc_id}.txt"

    # clean old content
    with filepath.open("w+") as f:
        f.write("")

    try:
        article = Article(url, language="en")
        article.download()
        article.parse()

        text = clean_text(article.text)

        with filepath.open("w+") as f:
            f.write(text)

        return text

    except Exception as e:
        logger.warning(f"Subject {subject.name()} - Could not crawl {url}. Error: {e.args}")


def clean_text(text: str) -> str:
    """Some simple processes to clean the crawled text."""

    arr = []
    for line in re.compile(r'\n+').split(text):
        line = line.strip()
        if not line:
            continue
        line = re.compile(r'\[\d[\d,\- ]*\]').sub("", line)  # remove citations
        arr.append(line.strip())
    return "\n".join(arr)


def get_urls(subject: Synset) -> List[str]:
    """Get all URLs returned by the Bing API."""

    url_filepath = get_url_path(subject)
    with url_filepath.open() as f:
        urls = [line.strip() for line in f.readlines() if line.strip()]

    return urls
