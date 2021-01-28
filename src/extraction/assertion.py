import json
from collections import Counter
from typing import List, Optional, Any, Dict, Union

from nltk.corpus import wordnet as wn
from spacy import symbols
from spacy.tokens.span import Span
from spacy.tokens.token import Token

from extraction.facet import Facet
from extraction.supporting import PREPOSITIONS_ALLOWED_IN_PHRASES, PREPOSITION_EDGES, find_long_phrase, \
    complete_predicate, get_target, find_object, find_head_of_preposition_facet, ADVERB_EDGES, \
    normalize_subject_noun_chunk, finalize_object, is_noun_or_proper_noun
from helper.constants import REDUNDANT_PREDICATE_PREFIXES, REDUNDANT_PREDICATE_SUFFIXES

MAX_FACET_LENGTH = 5


class Assertion(object):
    def __init__(self, subj, verb, obj, full_subj=None, full_pred=None, full_obj=None, facets=None, is_synthetic=False):
        if facets is None:
            facets = []

        self.subj = subj
        self.verb = verb
        self.obj = obj
        self.facets = facets.copy()
        self.is_synthetic = is_synthetic

        # complete subject
        if full_subj is None:
            self.full_subj = find_long_phrase(self.subj)
        else:
            self.full_subj = full_subj

        # complete predicate
        if full_pred is None:
            self.full_pred = self.verb if self.is_synthetic else complete_predicate(self.verb, self.obj)
        else:
            self.full_pred = full_pred

        # complete object
        if full_obj is None:
            # if isinstance(self.obj, Token) and self.obj.pos == symbols.VERB:
            #     self.full_obj = self.obj.doc[self.obj.i:self.obj.i + 1]
            # else:
            #     self.full_obj = find_long_phrase(self.obj)
            self.full_obj = find_long_phrase(self.obj)
        else:
            self.full_obj = full_obj

        # extract facets
        # add prepositional facets connected to object
        if not self.is_synthetic and self.obj is not None:
            add_prepositional_facets_to_list(facet_list=self.facets, head=self.obj,
                                             prep_not_in=PREPOSITIONS_ALLOWED_IN_PHRASES)
        # add adverbs of predicate and object as facets
        if not self.is_synthetic:
            self.facets.extend(find_adverb_facets(self.verb))
        if self.obj is not None:  # 400 facets
            self.facets.extend(find_adverb_facets(self.obj))
            # "adverb + adjective + head noun": 200 facets
            if is_noun_or_proper_noun(self.obj):
                for adj in self.obj.lefts:
                    if adj.dep == symbols.amod:
                        self.facets.extend(find_adverb_facets(adj))

        # [HACK] fix the case of "in order to"
        self.fix_in_order_to()

        self.subpart_revised: bool = False

        self.doc_id = None
        doc = None
        if isinstance(self.obj, Span) or isinstance(self.obj, Token):
            doc = self.obj.doc
        elif isinstance(self.full_obj, Span) or isinstance(self.full_obj, Token):
            doc = self.full_obj.doc
        if doc is not None:
            self.doc_id = doc.user_data.get("doc_id", None)

    def __str__(self):
        return self.to_json_dict()

    def to_json_dict(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_dict(self, simplify: bool = False, urls: List[str] = None) -> Dict[str, Any]:
        if not simplify:
            return {
                "subject": str(self.full_subj) if self.full_subj is not None else None,
                "predicate": self.full_pred.strip('$') if isinstance(self.full_pred, str) else ' '.join(
                    [str(token) for token in self.full_pred]),
                "object": str(self.full_obj) if self.full_obj is not None else None,
                "facets": [
                    facet.to_dict() for facet in self.facets
                ],
                "source": {
                    "doc_id": self.doc_id,
                    "url": urls[self.doc_id] if urls is not None and self.doc_id is not None else None,
                    "in_sentence": get_sentence_source(self.full_subj, self.full_pred, self.full_obj, self.facets)
                }
                # "source": self.obj.sent if isinstance(self.obj, Span) else None
            }
        else:
            p = SimplifiedAssertion(self).to_dict(simplifies_object=True, urls=urls)
            p.update({
                "subject": str(self.full_subj)
            })
            return p

    def fix_in_order_to(self) -> None:
        to_be_removed = set()
        to_be_added = []
        for i, facet in enumerate(self.facets):
            if facet.connector is None or facet.statement_head is None:
                continue

            new_full_connector = facet.connector.doc[facet.connector.i: facet.connector.i + 3]
            if new_full_connector.lower_ != "in order to":
                continue

            order = facet.statement_head
            purpose_verb = get_target(order, symbols.acl)
            if purpose_verb is None:
                continue

            purpose_objects = find_object(purpose_verb)
            if len(purpose_objects) == 0:
                to_be_added.append(Facet(facet.connector, purpose_verb, is_adverb=False, full_statement=[purpose_verb],
                                         full_connector=new_full_connector))
            else:
                for p_obj in purpose_objects:
                    new_full_statement = [purpose_verb]
                    new_full_statement.extend([token for token in find_long_phrase(p_obj)])
                    to_be_added.append(
                        Facet(facet.connector, purpose_verb, is_adverb=False, full_statement=new_full_statement,
                              full_connector=new_full_connector))

            to_be_removed.add(i)
        self.facets = [facet for i, facet in enumerate(self.facets) if i not in to_be_removed]
        self.facets.extend(to_be_added)


def add_prepositional_facets_to_list(facet_list: List[Facet], head: Token, prep_not_in=None) -> None:
    """
    Find all prepositional facets connected to `head` and add them to `facet_list`.
    Prepositions should not be in `prep_not_in`, otherwise `head` should be an adjective.
    """
    if prep_not_in is None:
        prep_not_in = []

    for child in head.children:
        if (child.dep in PREPOSITION_EDGES) and (
                child.lower_ not in prep_not_in or (head.pos_ == 'ADJ' and child.lower_ != "than")):
            stmt_head = find_head_of_preposition_facet(child)
            add_facet_and_conjuncts_to_list(facet_list, child, stmt_head)
            for conj in child.conjuncts:  # also extract conjuncts
                conj_stmt_head = find_head_of_preposition_facet(conj)
                add_facet_and_conjuncts_to_list(facet_list, conj, conj_stmt_head)


def add_facet_and_conjuncts_to_list(facet_list: List[Facet], prep: Token, statement_head: Token) -> None:
    facet_list.append(Facet(connector=prep, statement_head=statement_head))

    # also extract conjuncts of statement head
    if statement_head is not None:
        facet_list.extend([Facet(prep, conj_head) for conj_head in statement_head.conjuncts])


def find_adverb_facets(word: Token) -> List[Facet]:
    return [Facet(connector=None, statement_head=adverb, is_adverb=True) for adverb in word.children if
            adverb.dep in ADVERB_EDGES and adverb.is_alpha]


class SimplifiedAssertion(object):
    def __init__(self, assertion: Assertion):
        self.original_assertion = assertion
        self.subj = normalize_subject_noun_chunk(assertion.full_subj)
        self.pred = simplify_predicate(assertion.full_pred)
        self.obj = assertion.full_obj
        # filter facets by length
        self.facets = [f for f in assertion.facets if
                       f.full_statement is None or len(f.full_statement) <= MAX_FACET_LENGTH]

        self.doc_id = self.original_assertion.doc_id

    def __hash__(self):
        return hash(self.get_triple_str().lower())

    def __eq__(self, value):
        return self.get_triple_str().lower() == value.get_facet_str().lower()

    def get_triple_str(self):
        return str(self.subj) + " " + self.pred + " " + str(self.obj)

    def to_dict(self, simplifies_object: bool = False, urls: List[str] = None) -> dict:
        return {
            'subject': str(self.subj),
            'predicate': str(self.pred),
            'object': finalize_object(self.obj) if (simplifies_object and self.obj is not None) else str(
                self.obj) if self.obj is not None else None,
            'facets': [
                facet.to_dict() for facet in self.facets
            ],
            "source": {
                "doc_id": self.doc_id,
                "url": urls[self.doc_id] if urls and self.doc_id is not None else None,
                "in_sentence": get_sentence_source(
                    self.original_assertion.full_subj if self.original_assertion is not None else None,
                    self.original_assertion.full_pred if self.original_assertion is not None else None,
                    self.obj,
                    self.facets
                ),
            }
        }


def get_sentence_source(subj, pred, obj, facets):
    sentence = None
    tokens = None

    obj_start = None
    obj_end = None
    obj_start_char = None
    obj_end_char = None

    pred_matches = []

    subj_start = None
    subj_end = None
    subj_start_char = None
    subj_end_char = None

    facets_matches = []

    if isinstance(obj, Span):
        sentence = obj.sent
    elif pred is not None and len(pred) > 0 and isinstance(pred[0], Token):
        sentence = pred[0].sent
    elif facets is not None and len(facets) > 0 and facets[0].full_statement is not None:
        sentence = facets[0].full_statement[0].sent

    if sentence is not None:
        tokens = [str(token) for token in sentence]

        if isinstance(obj, Span) and obj.sent == sentence:
            obj_start = obj.start - sentence.start
            obj_end = obj.end - sentence.start
            obj_start_char = obj.start_char - sentence.start_char
            obj_end_char = obj.end_char - sentence.start_char

        if pred is not None and len(pred) > 0 and isinstance(pred[0], Token) and pred[0].sent == sentence:
            for pred_tok in pred:
                pred_matches.append({
                    # "token": pred_tok.text,
                    "start": pred_tok.i - sentence.start,
                    "end": pred_tok.i + 1 - sentence.start,
                    "start_char": pred_tok.idx - sentence.start_char,
                    "end_char": pred_tok.idx + len(pred_tok) - sentence.start_char,
                })

        if isinstance(subj, Span) and subj.sent == sentence:
            subj_start = subj.start - sentence.start
            subj_end = subj.end - sentence.start
            subj_start_char = subj.start_char - sentence.start_char
            subj_end_char = subj.end_char - sentence.start_char

        if facets is not None:
            for facet in facets:
                matches = []
                facet_tokens = []
                if facet.full_connector is not None:
                    facet_tokens.extend(token for token in facet.full_connector)
                if facet.full_statement is not None:
                    facet_tokens.extend(token for token in facet.full_statement)
                for token in facet_tokens:
                    matches.append({
                        "start": token.i - sentence.start,
                        "end": token.i + 1 - sentence.start,
                        "start_char": token.idx - sentence.start_char,
                        "end_char": token.idx + len(token) - sentence.start_char,
                    })
                facets_matches.append({
                    "matches": matches
                })

    return {
        "sentence": str(sentence),
        "tokens": tokens,
        "subj_start": subj_start,
        "subj_end": subj_end,
        "subj_start_char": subj_start_char,
        "subj_end_char": subj_end_char,
        "pred_matches": pred_matches,
        "obj_start": obj_start,
        "obj_end": obj_end,
        "obj_start_char": obj_start_char,
        "obj_end_char": obj_end_char,
        "facets_matches": facets_matches,
    }


class SimplifiedSubpartAssertion(SimplifiedAssertion):
    # noinspection PyMissingConstructor
    def __init__(self, subject_name: str, subj: str, pred: Optional[List[Token]], obj: Span, facets=None,
                 subj_root: Span = None):
        if facets is None:
            facets = []
        self.subject_name = subject_name
        self.subj = subj
        self.pred = simplify_predicate(pred)
        self.obj = obj
        self.facets = [facet for facet in facets if
                       facet.full_statement is None or len(facet.full_statement) <= MAX_FACET_LENGTH]

        self.doc_id = self.obj.doc.user_data.get("doc_id", None)
        self.subj_root = subj_root
        self.pred_root = pred

    def to_dict(self, simplifies_object: bool = False, urls: List[str] = None) -> dict:
        return {
            'subject': str(self.subj),
            'predicate': str(self.pred),
            'object': finalize_object(self.obj) if (simplifies_object and self.obj is not None) else str(
                self.obj) if self.obj is not None else None,
            'facets': [
                facet.to_dict() for facet in self.facets
            ],
            "source": {
                "doc_id": self.doc_id,
                "url": urls[self.doc_id] if urls and self.doc_id is not None else None,
                "in_sentence": get_sentence_source(
                    self.subj_root,
                    self.pred_root,
                    self.obj,
                    self.facets
                ),
            }
        }


class SameObjectAssertion(object):
    def __init__(self, simplified_assertion_list: List[SimplifiedAssertion]):
        simplified_assertion_list = sorted(simplified_assertion_list, key=lambda x: len(str(x.obj)))

        self.subj = simplified_assertion_list[0].subj
        self.pred = simplified_assertion_list[0].pred
        self.obj = find_representative_object(
            [simplified_assertion.obj for simplified_assertion in simplified_assertion_list])
        self.facets = Counter([f for a in simplified_assertion_list for f in a.facets])

        self.count = len(simplified_assertion_list)
        self.simplified_assertion_list = simplified_assertion_list
        self.object_list = [a.obj for a in simplified_assertion_list]


def find_representative_object(object_list: List[Span]) -> Optional[Span]:
    """Extract shortest one among the most occurring objects."""

    if len(object_list) == 0:
        return None

    string_2_object_list = {}
    for obj in object_list:
        string = obj.lower_
        if string in string_2_object_list:
            string_2_object_list[string].append(obj)
        else:
            string_2_object_list[string] = [obj]
    arr = sorted([s for s in string_2_object_list], key=lambda s: (-len(string_2_object_list[s]), len(s)))

    return string_2_object_list[arr[0]][0]


class SamePredicateAssertion(object):
    def __init__(self, same_object_assertion_list: List[SameObjectAssertion]):
        same_object_assertion_list = sorted(same_object_assertion_list, key=lambda x: len(str(x.obj)))
        self.subj = same_object_assertion_list[0].subj
        self.obj = same_object_assertion_list[0].obj
        self.pred_list = sorted([(assertion.pred, assertion.count) for assertion in same_object_assertion_list],
                                key=lambda x: -x[1])
        self.facets = Counter(
            [facet for same_object_assertion in same_object_assertion_list
             for simplified_assertion in same_object_assertion.simplified_assertion_list
             for facet in simplified_assertion.facets]
        )
        self.count = sum(x[1] for x in self.pred_list)
        self.same_object_assertion_list = same_object_assertion_list


def simplify_predicate(predicate: Union[str, List[Token]]) -> str:
    if isinstance(predicate, str):
        return predicate.strip("$")

    arr = [t for t in predicate]
    while len(arr) > 1 and arr[0].dep == symbols.aux:  # exclude modal verb
        arr = arr[1:]
    if arr[0].lemma_ == "be":
        arr[0] = arr[0].vocab[arr[0].lemma]
    elif arr[0].tag_ == "VBN":
        # past participle
        if (arr[0].dep == symbols.acl
                or arr[0].dep == symbols.advcl
                or any(gc.dep == symbols.auxpass for c in arr[0].conjuncts for gc in c.children)):
            arr.insert(0, arr[0].vocab["be"])
    elif arr[0].pos == symbols.VERB:
        arr[0] = arr[0].vocab[arr[0].lemma]
    elif len(arr) == 1:
        arr[0] = arr[0].vocab[arr[0].lemma]

    words = [t.lemma_ if (hasattr(t, 'dep') and t.dep == symbols.neg) else t.lower_ for t in arr]

    first = wn.morphy(words[0], wn.VERB)
    if first is not None:
        words[0] = first

    result = " ".join(words).strip()

    for prefix in REDUNDANT_PREDICATE_PREFIXES:
        prefix = prefix.strip() + " "
        if result.startswith(prefix):
            result = result[len(prefix):]
            break

    for suffix in REDUNDANT_PREDICATE_SUFFIXES:
        suffix = " " + suffix.strip()
        if result.endswith(suffix):
            result = result[:-len(suffix)]
            break

    return result.strip()
