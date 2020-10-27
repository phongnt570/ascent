from typing import List

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset

SYNSET2RULE = {
    "bird.n.01": "{} bird facts",
    "animal.n.01": "{} animal facts",
    "professional.n.01": "{} job description",
    "worker.n.01": "{} job description",
    "linguist.n.01": "{} job description",
    "entertainer.n.01": "{} job description",
    "capitalist.n.02": "{} job description",
    "engineer.n.01": "{} job description",
    "creator.n.02": "{} job description",
    "defender.n.01": "{} job description",
    "leader.n.01": "{} job description",
    "expert.n.01": "{} job description",
    "intellectual.n.01": "{} job description",
    "communicator.n.01": "{} job description",
    "official.n.01": "{} job description",
    "fiduciary.n.01": "{} job description",
    "person.n.01": "what does {} do?",
    "plant.n.02": "{} plant facts",
    "beverage.n.01": "{} drink facts",
    "food.n.01": "{} food facts",
    "food.n.02": "{} food facts",
    "abstraction.n.01": "what is {}?",
    "abstraction.n.06": "what is {}?",
    "device.n.01": "what is {}?",
    "material.n.01": "{} material facts",
    "drug.n.01": "{} drug facts",
    "tree.n.01": "{} tree facts",
    "vehicle.n.01": "{} vehicle facts",
    "clothing.n.01": "{} cloth facts",
    "medicine.n.02": "{} medicine facts",
    "phenomenon.n.01": "what is {}?",
    "sport.n.01": "{} sport facts",
    "weapon.n.01": "{} weapon facts",
    "instrumentality.n.03": "what is {}?",
    "location.n.01": "{} (location) facts",
}

SYNSET2RULE = {wn.synset(k): v for k, v in SYNSET2RULE.items()}


def hyper(s: Synset) -> List[Synset]:
    return s.hypernyms()


def get_search_query(subject: Synset) -> str:
    hypernym_set = set(subject.closure(hyper))

    template = ""
    shortest_distance = 20
    for domain in SYNSET2RULE:
        if domain in hypernym_set:
            distance = subject.shortest_path_distance(domain)
            if distance < shortest_distance:
                template = SYNSET2RULE[domain]
                shortest_distance = distance

    lemma: str = subject.lemma_names()[0].replace("_", " ")

    if template:
        return template.format(lemma)

    if subject.hypernyms():
        hypernym = subject.hypernyms()[0].lemma_names()[0].replace("_", " ")
        if hypernym not in lemma:
            return f"{lemma} ({hypernym})"

    return lemma


def get_wikipedia_search_query(subject: Synset) -> str:
    lemma: str = subject.lemma_names()[0].replace("_", " ")
    query: str = lemma

    hypernym_set = set(subject.closure(hyper))

    if wn.synset("animal.n.01") in hypernym_set:
        query = f"{lemma} (animal)"

    elif wn.synset("person.n.01") in hypernym_set:
        query = f"{lemma} (person)"

    elif wn.synset("plant.n.02") in hypernym_set:
        query = f"{lemma} (plant)"

    elif wn.synset("beverage.n.01") in hypernym_set:
        query = f"{lemma} (drink)"

    elif wn.synset("food.n.01") in hypernym_set or wn.synset("food.n.02") in hypernym_set:
        query = f"{lemma} (food)"

    return query + " site:en.wikipedia.org"


def has_hypernym(synset: Synset, other: Synset) -> bool:
    return other in set(synset.closure(hyper))
