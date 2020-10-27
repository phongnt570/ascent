import json
from typing import List

from spacy import symbols
from spacy.tokens import Token

from helper.constants import SPECIAL_FACET_CONNECTORS
from extraction.supporting import get_span, find_long_phrase, get_target, find_head_of_preposition_facet, find_object, \
    complete_predicate, PREPOSITION_EDGES

# to complete facet connectors
EDGES_ALLOWED_IN_FACET_CONNECTORS = {symbols.amod}


class Facet(object):
    def __init__(self, connector, statement_head, is_adverb=False, full_statement=None, full_connector=None,
                 is_purpose=False):
        self.connector = connector  # can be preposition (e.g., in, on, at) or None (for adverb facet)
        self.statement_head = statement_head
        self.is_adverb = is_adverb
        self.is_purpose = is_purpose

        self.full_connector = full_connector if full_connector is not None else complete_connector(self.connector)
        self.full_statement = full_statement

        if full_statement is None:
            if self.is_adverb:
                self.full_statement = complete_adverb_facet(self.statement_head)
            elif self.statement_head is not None:
                self.full_statement = find_long_phrase(self.statement_head, prep_in=set())
            else:
                self.full_statement = find_statement_for_non_objective_prep(self.connector)

    def __hash__(self):
        return hash(str(self).lower())

    def __eq__(self, value):
        return str(self).lower() == str(value).lower()

    def __str__(self):
        return self.to_json_dict()

    def to_json_dict(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_dict(self):
        return {
            "connector": None if self.full_connector is None else str(self.full_connector),
            "statement": "".join([t.text_with_ws for t in self.full_statement]).strip() if (
                    self.full_statement is not None) else None,
        }

    def get_tokens(self) -> List[Token]:
        tokens = []
        if self.full_connector is not None:
            tokens.extend([t for t in self.full_connector])
        if self.full_statement is not None:
            tokens.extend([t for t in self.full_statement])
        return tokens

    def get_text(self) -> str:
        return "".join([token.text_with_ws for token in self.get_tokens()]).strip()


def complete_connector(prep):
    if prep is None:
        return None
    phrase = [prep]
    for child in prep.children:
        if child.dep in EDGES_ALLOWED_IN_FACET_CONNECTORS and (child.i == prep.i - 1 or child.i == prep.i + 1):
            phrase.append(child)
            break
    try:  # [HACK] "due to", "because of",...
        if (prep.lower_ + " " + prep.doc[prep.i + 1].lower_) in SPECIAL_FACET_CONNECTORS:
            phrase.append(prep.doc[prep.i + 1])
    except IndexError:
        pass

    return get_span(phrase)


def complete_adverb_facet(adverb):
    if adverb is None:
        return None
    phrase = [adverb]
    for child in adverb.children:
        if child.dep != symbols.conj and child.dep != symbols.cc:
            if child.dep in PREPOSITION_EDGES:
                phrase.append(child)
                x = find_head_of_preposition_facet(child)
                if x is not None:
                    phrase.extend([token for token in find_long_phrase(x)])
            else:
                phrase.extend(find_long_phrase(child))
    return get_span(phrase)


def find_statement_for_non_objective_prep(prep):
    # multiple prep, e.g "compared --(prep)-- to ..."
    prep_list = [child for child in prep.children if child.dep == symbols.prep]
    if len(prep_list) > 0:
        next_prep = prep_list[0]
        stmt_head = find_head_of_preposition_facet(next_prep)
        if stmt_head is not None:
            np = find_long_phrase(stmt_head, prep_in=set())
            if next_prep.i < np[0].i:
                return np.doc[next_prep.i:np[-1].i + 1]

    # pcomp, e.g. "for --(pcomp)-- circling ..."
    pcomp_list = [child for child in prep.children if child.dep == symbols.pcomp]
    if len(pcomp_list) > 0:
        verb = pcomp_list[0]
        if verb.pos == symbols.VERB:
            obj = find_object(verb)
            if len(obj) > 0:
                obj = obj[0]
                np = find_long_phrase(obj, prep_in=set())
                verb = complete_predicate(verb)
                return np.doc[verb[0].i:np[-1].i + 1]
            else:
                verb_prep = get_target(verb, symbols.prep)
                if verb_prep is not None:
                    pobj = get_target(verb_prep, symbols.pobj)
                    if pobj is not None:
                        np = find_long_phrase(pobj, prep_in=set())
                        verb = complete_predicate(verb)
                        return np.doc[verb[0].i:np[-1].i + 1]
