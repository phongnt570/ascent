"""Implement StuffIE and supporting functions."""

from typing import List, Tuple, Set

from spacy import symbols
from spacy.language import Language
from spacy.tokens import Doc
from spacy.tokens.span import Span
from spacy.tokens.token import Token

from extraction.assertion import Assertion, add_prepositional_facets_to_list
from extraction.facet import Facet
from extraction.supporting import find_object, get_target, get_conjunctions, TOBE, find_long_phrase, get_span, \
    PREPOSITION_EDGES, is_comparative_adj, ADVERBS_ALLOWED_IN_PHRASES, is_special_adverb
from helper.constants import SPECIAL_PREDICATES, IGNORED_ADVERB_FACETS, IGNORED_FACET_PREFIXES
import inflect

# to find subject
EDGES_OF_SUBJECTS = {symbols.nsubj, symbols.nsubjpass, symbols.csubj}

# to find missing subject
EDGES_OF_MISSING_SUBJECTS = {symbols.advcl, symbols.xcomp}  # , symbols.pcomp

RELATIVE_CLAUSES = {"that", "which", "what", "who", "whom", "where", "when", "why"}

# inflect
INFLECT_ENGINE = inflect.engine()


def get_plural_form_of_noun_phrase(noun_phrase: str) -> str:
    words = noun_phrase.split()
    words[-1] = INFLECT_ENGINE.plural_noun(words[-1])
    return " ".join(words)


def run_extraction(lines: List[str], doc_ids: List[int], spacy_nlp: Language, subject: str) -> Tuple[List[Doc], List[Assertion], int]:
    """This function is called by the extractor pipeline and
    returns also list of processed documents and number of sentences (for statistic purpose)"""

    doc_list: List[Doc] = []
    assertion_list: List[Assertion] = []
    cnt_sent: int = 0

    subject_less_idx: Set[int] = set()

    for line, doc_id in zip(lines, doc_ids):
        if len(line.split()) > 1000:  # ignore huge paragraphs
            continue

        doc: Doc = spacy_nlp(line.strip())
        doc.user_data["doc_id"] = doc_id
        doc_list.append(doc)

        extracted = 0
        for sent in doc.sents:
            s = Stuffie(sent)
            s.parse()
            assertion_list.extend(s.assertions)
            cnt_sent += 1
            extracted += len(s.assertions)

        # [HACK] fix single-sentence paragraph with no subjects
        # TODO maybe more proper filters
        if extracted == 0 and len(list(doc.sents)) == 1 and len(doc) >= 5:
            if doc[0].pos != symbols.VERB:
                continue
            if doc[0].lower_ in {"view", "click", "see", "discover", "find", "explore", "search"}:
                continue
            subject_less_idx.add(len(doc_list) - 1)

        # stop early to avoid time-exploding
        if cnt_sent > 50000:
            break

    for idx in subject_less_idx:
        # subject-less sentences should be a list of other structure-similar sentences
        if (idx + 1) not in subject_less_idx and (idx - 1) not in subject_less_idx:
            continue

        doc = doc_list[idx]

        if (idx + 1) in subject_less_idx and doc[:3].lower_ == doc_list[idx + 1][:3].lower_:
            continue
        if (idx - 1) in subject_less_idx and doc[:3].lower_ == doc_list[idx - 1][:3].lower_:
            continue

        plural_subject = get_plural_form_of_noun_phrase(subject).capitalize()
        verb = doc[0].lemma_.lower()
        new_line = plural_subject + " " + verb + " " + str(doc[1:])

        new_doc = spacy_nlp(new_line)
        s = Stuffie(list(new_doc.sents)[0])
        s.parse()
        assertion_list.extend(s.assertions)

        doc_list[idx] = new_doc

    return doc_list, assertion_list, cnt_sent


def run_stuffie(text: str, spacy_nlp: Language, do_eval=False) -> List[Assertion]:
    """Single run of StuffIE, independent of DESCENT project"""

    assertion_list = []

    doc = spacy_nlp(text.strip())
    for sent in doc.sents:
        stuffie = Stuffie(sent, do_eval)
        stuffie.parse()
        assertion_list.extend(stuffie.assertions)
    return assertion_list


class Stuffie(object):
    """Implementation of StuffIE algorithm (with modifications)"""

    def __init__(self, sentence: Span, do_eval: bool = False):
        self.sentence: Span = sentence
        self.assertions: List[Assertion] = []
        self.do_eval = do_eval

    def parse(self):
        """Extract assertions in the sentence given to this Stuffie object through initialization."""

        verbs = [token for token in self.sentence if token.pos == symbols.VERB]

        # verb predicates
        for verb in verbs:
            # find subjects
            subject_list = find_subject(verb)
            if len(subject_list) == 0:
                subject_list = find_missing_subject(verb)
            if len(subject_list) == 0:
                continue

            # find objects
            object_list = find_object(verb)
            if len(object_list) == 0:
                object_list = [None]

            # find facets
            facets = find_facets(verb)

            # add to assertion list
            for subj in subject_list:
                for obj in object_list:
                    asst = Assertion(subj, verb, obj, facets=facets)
                    if obj is not None:
                        self.assertions.append(asst)
                    else:  # revise None object
                        self.assertions.extend(revise_none_object_assertion(asst))

        # [HACK] fix long predicates
        self.fix_long_predicates()

        # [HACK] Hearst patterns: "such as", "like", "including"
        self.extract_examples()

        # [HACK] fix the case of "be able to"
        self.extract_be_able_to()

        # [HACK] special predicates: be + adj + preposition, e.g.: "be capable/incapable of", "be responsible for"
        self.extract_special_predicates()

        # apposition relations
        self.assertions.extend(self.find_appos_relations())

        # filter out unwanted assertions
        self.assertions = filter_assertion_list(self.assertions)

        # sort the assertions by subject index
        self.assertions = sorted(self.assertions, key=lambda a: a.subj.i)

        # filter out unwanted facets
        if not self.do_eval:
            for assertion in self.assertions:
                assertion.facets = filter_facets(assertion.facets)
        else:
            for assertion in self.assertions:
                assertion.facets = list(
                    set(facet for facet in assertion.facets if not is_special_adverb(facet.statement_head)))

    def find_appos_relations(self) -> List[Assertion]:
        assertion_list = []
        for token in self.sentence:
            if token.dep == symbols.appos:
                assertion_list.append(Assertion(token.head, TOBE, token, is_synthetic=True))
        return assertion_list

    def fix_long_predicates(self):
        to_be_removed = set()
        for i in range(len(self.assertions)):
            if i in to_be_removed:
                continue
            cur = self.assertions[i]
            for j in range(i + 1, min(len(self.assertions), i + 3)):
                if j in to_be_removed:
                    continue
                nex = self.assertions[j]

                if cur.subj == nex.subj:
                    aux = get_target(nex.verb, symbols.aux)
                    # merge <; have adapted to live; >, <; to live; >
                    if get_target(cur.verb, symbols.xcomp) == nex.verb:
                        if (cur.obj is None and nex.obj is None) or (cur.obj == nex.obj):
                            cur.facets.extend([facet for facet in nex.facets if facet not in cur.facets])
                            to_be_removed.add(j)
                            break
                        elif cur.obj is None and nex.obj is not None:
                            new_full_pred = set([t for t in cur.full_pred])
                            new_full_pred.update(set([t for t in nex.full_pred]))
                            new_full_pred = sorted(new_full_pred, key=lambda x: x.i)
                            nex.full_pred = new_full_pred
                            to_be_removed.add(i)
                        elif cur.obj is not None and nex.obj is None:
                            to_be_removed.add(j)
                    # verb facet
                    if (
                            aux is not None and
                            (
                                    get_target(cur.verb, symbols.advcl) == nex.verb
                                    or (
                                            get_target(cur.verb, symbols.xcomp) == nex.verb
                                            and cur.obj is not None and nex.obj is not None and cur.obj != nex.obj
                                    )
                            )
                    ):
                        if aux.lower_ == "to" or aux.lower_ == "for":
                            new_full_statement = [t for t in nex.full_pred if t != aux]
                            if nex.full_obj is not None:
                                new_full_statement.extend([t for t in nex.full_obj])
                                new_full_statement = sorted(new_full_statement, key=lambda x: x.i)
                            cur.facets.append(
                                Facet(aux, None, False, full_statement=new_full_statement, is_purpose=True))
                            to_be_removed.add(j)

        self.assertions = [self.assertions[i] for i in range(len(self.assertions)) if i not in to_be_removed]

    def extract_examples(self):
        to_be_removed = set()
        new_assertion_list = []
        for i, asst in enumerate(self.assertions):
            if asst.obj is None:
                continue
            such_as_fcs = [fc for fc in asst.facets
                           if fc.statement_head is not None
                           and fc.connector is not None
                           and fc.connector.head == asst.obj
                           and fc.full_connector.lower_ in ["such as", "like", "including"]
                           ]
            if not such_as_fcs:
                continue
            new_facets = [fc for fc in asst.facets if fc not in set(such_as_fcs)]
            for fc in such_as_fcs:
                new_obj = fc.statement_head
                new_full_obj = fc.full_statement
                new_assertion_list.append(
                    Assertion(asst.subj, asst.verb, new_obj, asst.full_subj, asst.full_pred, new_full_obj, new_facets))
            to_be_removed.add(i)
        self.assertions = [self.assertions[i] for i in range(len(self.assertions)) if i not in to_be_removed]
        self.assertions.extend(new_assertion_list)

    def extract_be_able_to(self):
        candidates = []
        for asst in self.assertions:
            if asst.obj is not None and asst.obj.pos_ == "ADJ":
                xc = get_target(asst.obj, symbols.xcomp)
                if xc is not None:
                    ax = get_target(xc, symbols.aux)
                    if ax is not None and ax.lower_ == "to" and ax.i == asst.obj.i + 1:
                        candidates.append((asst, ax, xc))
        for asst, ax, xc in candidates:
            new_full_pred = [t for t in asst.full_pred]
            new_full_pred.extend([asst.obj, ax])
            new_obj = xc
            new_full_obj = [xc]
            x = find_object(xc)
            if len(x) > 0:
                new_full_obj.extend([t for t in find_long_phrase(x[0])])
            new_full_obj = get_span(new_full_obj)
            # update
            asst.full_pred = new_full_pred
            asst.obj = new_obj
            asst.full_obj = new_full_obj

    def extract_special_predicates(self):
        new_assertion_list = []
        to_be_removed = set()
        for i, asst in enumerate(self.assertions):
            if not asst.is_synthetic and asst.obj is not None:
                for facet in asst.facets:
                    new_full_pred = [t for t in asst.full_pred]
                    new_full_pred.append(asst.obj)
                    if facet.connector is not None and facet.connector.i == asst.obj.i + 1:
                        new_full_pred.append(facet.connector)
                        pred_text = asst.full_pred[
                                        -1].lemma_.lower() + ' ' + asst.obj.lower_ + ' ' + facet.connector.lower_
                        if (pred_text in SPECIAL_PREDICATES
                                or (asst.obj.pos_ == 'ADJ'
                                    and not is_comparative_adj(asst.obj)
                                    and facet.connector.lemma_.lower() == 'of')):
                            if facet.full_statement is not None:
                                new_facets = [old_facet for old_facet in asst.facets if
                                              old_facet.connector != facet.connector]
                                new_full_obj = get_span(facet.full_statement)
                                new_obj = new_full_obj.root
                                new_assertion_list.append(Assertion(
                                    subj=asst.subj,
                                    verb=asst.verb,
                                    obj=new_obj,
                                    full_subj=asst.full_subj,
                                    full_pred=new_full_pred,
                                    full_obj=new_full_obj,
                                    facets=new_facets
                                ))
                                to_be_removed.add(i)

        if to_be_removed and new_assertion_list:
            self.assertions = [self.assertions[i] for i in range(len(self.assertions)) if i not in to_be_removed]
            self.assertions.extend(new_assertion_list)


def find_subject(verb: Token) -> List[Token]:
    if verb is None:
        return []

    # relative clause
    #
    # ex1: "cat is an animal that is cute"
    # <cat; is; an animal>
    # <cat; is; cute>
    #
    # ex2: "a doctor is a person working at the hospital"
    # <a doctor; is; a person>
    # <a doctor; working at; the hospital>
    #
    if verb.dep in {symbols.relcl, symbols.acl}:  # verb = the second "is"
        potential = [subj for subj in verb.children if subj.dep in EDGES_OF_SUBJECTS]
        if len(potential) == 0 or potential[0].lower_ in RELATIVE_CLAUSES:
            if verb.head != verb and verb.head.pos == symbols.NOUN:  # verb.head = "animal"
                main_verb = verb.head.head  # main_verb = the first "is"
                if main_verb != verb.head:
                    if main_verb.lemma_ == "be":
                        return find_subject(main_verb)
                    else:
                        return [verb.head]

    # normal subjects
    # just like in StuffIE
    subject_list = []
    for child in verb.children:
        if child.dep in EDGES_OF_SUBJECTS:
            subject_list.append(child)
            subject_list.extend(get_conjunctions(child))  # also extract conjuncts
            return subject_list

    return []


def find_missing_subject(verb: Token) -> List[Token]:
    if verb.dep in EDGES_OF_MISSING_SUBJECTS:
        e = verb.dep
        head_verb = verb.head
        if head_verb.pos != symbols.VERB:
            if head_verb is None:
                return []
            head_verb = head_verb.head
            if head_verb is None or head_verb.pos != symbols.VERB:
                return []
        mark = get_target(verb, symbols.mark)
        if mark is not None:
            mark = mark.lower_

        if ((e == symbols.xcomp and mark == "to")
                or (e == symbols.advcl and mark == "for")):
            ms = find_object(head_verb)
        else:
            ms = find_subject(head_verb)
            if len(ms) == 0:
                ms = find_missing_subject(head_verb)
        if len(ms) > 0:
            return ms

    # also extract subjects of conjunct verbs
    if verb.dep == symbols.conj:
        ms = find_subject(verb.head)
        if len(ms) == 0:
            ms = find_missing_subject(verb.head)
        return ms
    return []


def find_facets(verb: Token) -> List[Facet]:
    facet_list = []
    add_prepositional_facets_to_list(facet_list, verb)
    return facet_list


def revise_none_object_assertion(none_object_assertion: Assertion) -> List[Assertion]:
    # find the facet which will be changed to object
    candidates = [facet for facet in none_object_assertion.facets
                  if facet.connector is not None
                  and facet.connector.lower_ not in {"because", "due", "than", "beneath", "between", "from", "during",
                                                     "both", "despite"}
                  and facet.connector.head == none_object_assertion.verb
                  and facet.connector.dep in PREPOSITION_EDGES
                  and facet.statement_head is not None
                  and not isinstance(facet.full_statement, list)
                  and not has_entity_type(facet.full_statement, symbols.TIME)
                  ]
    if not candidates:
        predicate = none_object_assertion.full_pred
        if len(predicate) >= 2 and predicate[-1] == none_object_assertion.verb \
                and get_target(predicate[-1], symbols.auxpass) in predicate:
            return [(Assertion(subj=none_object_assertion.subj,
                               verb=predicate[0],
                               obj=predicate[-1],
                               full_subj=none_object_assertion.full_subj,
                               full_pred=predicate[:-1],
                               full_obj=predicate[0].doc[predicate[-1].i:(predicate[-1].i + 1)],
                               facets=none_object_assertion.facets))]

        return [none_object_assertion]

    candidates = sorted(candidates,
                        key=lambda x: (
                            not (none_object_assertion.verb.lemma_.lower() in {"go", "live", "survive"}
                                 and x.connector.lower_ == "without"),
                            not has_entity_type(x.full_statement, symbols.LOC),
                            x.connector.i < none_object_assertion.verb.i,
                            abs(x.connector.i - none_object_assertion.verb.i)
                        )
                        )
    facet = candidates[0]
    results = []
    facet_conjuncts = [ff for ff in none_object_assertion.facets
                       if ff.connector is not None
                       and ff.connector.head == none_object_assertion.verb
                       and ff.connector.lower_ == facet.connector.lower_
                       and ff.statement_head is not None
                       and not isinstance(ff.full_statement, list)
                       ]

    new_facets = [ff for ff in none_object_assertion.facets if ff not in set(facet_conjuncts)]
    new_full_pred = none_object_assertion.full_pred.copy()
    new_full_pred.append(facet.connector)

    for ff in facet_conjuncts:
        new_obj = ff.full_statement.root
        results.append(Assertion(subj=none_object_assertion.subj,
                                 verb=none_object_assertion.verb,
                                 obj=new_obj,
                                 full_subj=none_object_assertion.full_subj,
                                 full_pred=new_full_pred,
                                 full_obj=None,
                                 facets=new_facets
                                 ))

    if not results:
        return [none_object_assertion]

    return results


def has_entity_type(span: Span, ent_type) -> bool:
    return any([token.ent_type == ent_type for token in span])


def filter_assertion_list(assertion_list: List[Assertion]) -> List[Assertion]:
    to_be_removed = set()
    for ind, assertion in enumerate(assertion_list):
        subject = assertion.subj
        if 0 < subject.i < len(subject.doc) - 1:
            doc = subject.doc
            i = subject.i
            if doc[(i - 1):(i + 2)].lower_ == 'in order to':
                to_be_removed.add(ind)

    return [assertion_list[ind] for ind, _ in enumerate(assertion_list) if ind not in to_be_removed]


def filter_facets(facet_list: List[Facet]) -> List[Facet]:
    """
    Filter out unwanted facets which are listed in two files:
    `ignored_adverbs_facets.txt` and `ignored_facet_prefixes.txt`.
    """
    facet_list = list(set(facet for facet in facet_list if
                          facet.get_text().lower() not in IGNORED_ADVERB_FACETS.union(
                              ADVERBS_ALLOWED_IN_PHRASES) and not is_special_adverb(facet.statement_head)))

    new_facet_list: List[Facet] = []
    for facet in facet_list:
        facet_text = facet.get_text().lower()

        flag = False
        for prefix in IGNORED_FACET_PREFIXES:
            if facet_text.startswith(prefix):
                flag = True
                break
        if flag:
            continue

        if facet.connector is None and facet.full_statement is not None:
            tokens_of_facet = facet.get_tokens()
            if any([token.like_num for token in tokens_of_facet]):
                continue
            if len(tokens_of_facet) == 1 and tokens_of_facet[0].pos in {symbols.NOUN, symbols.ADJ}:
                continue

        if facet.full_statement is None:
            continue

        new_facet_list.append(facet)

    return new_facet_list
