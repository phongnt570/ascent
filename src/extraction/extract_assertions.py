"""Contains functions to extract assertions relevant to a given subject, which include general assertions,
subgroup assertions and assertions of associated terms."""

from typing import List, Tuple, Dict

from spacy import symbols
from spacy.tokens.span import Span
from spacy.tokens.token import Token

from extraction.assertion import SimplifiedAssertion, SimplifiedSubpartAssertion, find_adverb_facets, Assertion
from extraction.extract_terms import Subgroup, Subpart
from extraction.supporting import finalize_object, is_comparative_adj, find_compound_noun, \
    normalize_subject_noun_chunk, get_target, find_short_phrase, TOBE, ADVERB_EDGES, get_span, get_conjunctions, \
    chunk_ends_with_tokens, is_noun_or_proper_noun, is_adjective, is_special_adverb
from helper.constants import ALL_PRONOUNS, ONE_WORD_OBJECT_IGNORED, IGNORED_PREDICATES, PRONOUNS_IGNORED

MAX_OBJ_LEN = 30
ONE_WORD_OBJ_POS_IGNORED = {symbols.DET, symbols.PRON, symbols.ADV, symbols.ADP}


def extract_general_and_subgroup_assertions(assertion_list: List[Assertion],
                                            subject: str,
                                            subgroup_list: List[Subgroup]) \
        -> Tuple[List[SimplifiedAssertion], List[SimplifiedAssertion]]:
    """Extract general and subgroup assertions from the full list of OpenIE assertions."""

    # head noun of subject, used to retrieve possible relevant co-reference
    subject_head_noun = subject.lower().split()[-1]

    # subgroup mapping
    phrase2subgroup_name = get_phrase_mapping(subgroup_list)

    # resulting lists
    general_assertions = []
    subgroup_assertions = []

    # check each assertion candidate
    for assertion in assertion_list:
        # pre-checking phase
        if not pre_check_assertion(assertion):
            continue

        # ignore irrelevant subject
        if not is_relevant_subject(assertion.full_subj, subject_head_noun):
            continue

        # save for restore
        original_full_subj = assertion.full_subj

        # take care of co-reference subjects
        if is_coref_relevant_subject(assertion.full_subj, subject_head_noun):
            assertion.full_subj = get_coref_main(assertion.full_subj)

        # simplify subject and predicate
        simplified_assertion = SimplifiedAssertion(assertion)

        # restore the original assertion
        assertion.full_subj = original_full_subj

        # push the assertion to correct resulting list
        if simplified_assertion.subj == subject:  # general assertion
            general_assertions.append(simplified_assertion)
        elif simplified_assertion.subj in phrase2subgroup_name:  # subgroup assertion
            # change subgroup phrase to its representative
            simplified_assertion.subj = phrase2subgroup_name[simplified_assertion.subj]
            subgroup_assertions.append(simplified_assertion)

    # sort the lists
    general_assertions = sorted(filter_assertion_list(general_assertions), key=lambda x: (x.subj, x.pred, str(x.obj)))
    subgroup_assertions = sorted(filter_assertion_list(subgroup_assertions), key=lambda x: (x.subj, x.pred, str(x.obj)))

    return general_assertions, subgroup_assertions


def extract_subpart_assertions(assertion_list: List[Assertion], target_subject: str, subgroup_list: List[Subgroup],
                               subpart_list: List[Subpart]) -> List[SimplifiedAssertion]:
    """Extract assertions of the associated terms from the full list of OpenIE assertions and SpaCy's noun chunks."""

    # head noun of subject, used to retrieve possible relevant co-reference
    tokens_of_subject = target_subject.lower().split()
    subject_head_noun = tokens_of_subject[-1]

    # set of subpart strings
    subpart_string_set = set([subpart.name for subpart in subpart_list])

    # subgroup mapping
    phrase2subgroup_name = get_phrase_mapping(subgroup_list)

    # resulting list
    subpart_assertions = []

    # to avoid duplicates
    revised_possessives = set()

    # from extracted assertions
    for assertion in assertion_list:
        if assertion.subpart_revised:
            continue
        # pre-checking phase
        if not pre_check_assertion(assertion):
            continue

        # save to restore
        original_full_subj = assertion.full_subj

        # check if the current assertion is a valid candidate
        is_candidate = False
        subject_is_subpart = False

        # subject is subpart
        if find_compound_noun(assertion.subj) in subpart_string_set:
            is_candidate = True
            subject_is_subpart = True

        # object is subpart
        elif find_compound_noun(assertion.obj) in subpart_string_set:
            if is_relevant_subject(assertion.full_subj, subject_head_noun):
                assertion_subject = assertion.full_subj
                if is_coref_relevant_subject(assertion.full_subj, subject_head_noun):
                    assertion_subject = get_coref_main(assertion.full_subj)
                simplified_subject = normalize_subject_noun_chunk(assertion_subject)
                if simplified_subject in phrase2subgroup_name or simplified_subject == target_subject:
                    assertion.full_subj = assertion_subject
                    is_candidate = True
                    subject_is_subpart = False

        # extract subpart assertion
        if is_candidate:
            # subject is subpart
            if subject_is_subpart:
                # find the possessive
                possessive = get_target(assertion.full_subj.root, symbols.poss)

                # extract subject/subgroup name
                subject_name = ""
                # <possessive + subpart ; predicate ; object>
                # possessive is only used for finding the subgroup name
                if possessive is not None:
                    # possessive is subject or subgroup, e.g. <lynx's short tail; help; it>
                    if chunk_ends_with_tokens(find_short_phrase(possessive), tokens_of_subject):
                        subject_name = normalize_subject_noun_chunk(find_short_phrase(possessive))
                    # possessive pronoun refers to subject/subgroup, e.g. <their short tail; help; it>
                    elif refers_to_subject_or_subgroups(possessive, subject_head_noun):
                        subject_name = normalize_subject_noun_chunk(get_coref_clusters(possessive)[0].main)
                    # double-check
                    if subject_name not in phrase2subgroup_name and subject_name != target_subject:
                        subject_name = ""

                # could not find coref of possessive, then get the nearest one
                if possessive is not None and not subject_name:
                    subject_name = get_nearest_before_subject_name(assertion.subj, target_subject,
                                                                   phrase2subgroup_name)
                    # still could not find anything
                    if not subject_name:
                        continue

                # find correct representative
                if subject_name in phrase2subgroup_name:  # subgroup
                    subject_name = phrase2subgroup_name[subject_name]
                else:  # target subject
                    subject_name = target_subject

                # find subpart string
                subj_subpart = find_compound_noun(assertion.full_subj.root)

                # discard if it's not in the subpart set
                if subj_subpart not in subpart_string_set:
                    continue

                # push it to the result
                subpart_assertions.append(SimplifiedSubpartAssertion(subject_name=subject_name,
                                                                     subj=subj_subpart,
                                                                     pred=assertion.full_pred,
                                                                     obj=assertion.full_obj,
                                                                     facets=assertion.facets))
                assertion.subpart_revised = True

            # object is subpart: <subject/subgroup ; predicate ; adj + subpart> -> <subpart ; be ; adj>
            else:
                # extract subject/subgroup name
                subject_name = normalize_subject_noun_chunk(assertion.full_subj)
                if subject_name != target_subject and subject_name not in phrase2subgroup_name:
                    continue
                if subject_name in phrase2subgroup_name:
                    subject_name = phrase2subgroup_name[subject_name]

                # extract subpart name
                subj_subpart = find_compound_noun(assertion.obj)
                if subj_subpart not in subpart_string_set:
                    continue

                # get all adjectives supporting the subpart
                object_list = extract_characteristics_of_subpart(assertion.full_obj)
                for obj in object_list:
                    # extract adverb facets
                    facets = find_adverb_facets(obj.root)
                    # push it to the result
                    subpart_assertions.append(SimplifiedSubpartAssertion(subject_name=subject_name,
                                                                         subj=subj_subpart,
                                                                         pred=TOBE,
                                                                         obj=obj,
                                                                         facets=facets))

                # add the possessives to the revised set
                revised_possessives.update([t.dep == symbols.poss for t in assertion.full_obj])
                assertion.subpart_revised = True

        # restore the original assertion
        assertion.full_subj = original_full_subj

    # from chunk pattern: `possessive + adj + subpart`
    for subpart in subpart_list:
        candidates = [chunk for chunk in subpart.chunk_list if
                      any([(t.dep == symbols.poss and t not in revised_possessives) for t in chunk])]

        for noun_chunk in candidates:
            possessive = [t for t in noun_chunk if t.dep == symbols.poss][0]
            subject_name = ""

            # possessive is subject or subgroup, e.g. "lynx's short tail"
            if chunk_ends_with_tokens(find_short_phrase(possessive), tokens_of_subject):
                subject_name = normalize_subject_noun_chunk(find_short_phrase(possessive))
            # possessive pronoun refers to subject/subgroup, e.g. "their short tail"
            elif refers_to_subject_or_subgroups(possessive, subject_head_noun):
                subject_name = normalize_subject_noun_chunk(get_coref_clusters(possessive)[0].main)

            # double-check
            if subject_name != target_subject and subject_name not in phrase2subgroup_name:
                continue

            # extract all adjectives supporting the subpart
            object_list = extract_characteristics_of_subpart(noun_chunk)
            for obj in object_list:
                # extract adverb facets
                facets = find_adverb_facets(obj.root)
                # push it to the result
                subpart_assertions.append(SimplifiedSubpartAssertion(subject_name=subject_name,
                                                                     subj=subpart.name,
                                                                     pred=TOBE,
                                                                     obj=obj,
                                                                     facets=facets))

    # sort and filter out unwanted assertions
    subpart_assertions = sorted(filter_assertion_list(subpart_assertions),
                                key=lambda x: (x.subject_name, x.subj, x.pred, str(x.obj)))

    return subpart_assertions


def get_phrase_mapping(subgroup_list: List[Subgroup]) -> Dict[str, str]:
    """Map each subgroup phrase to its representative. For e.g., `the canadian lynx` -> `canada lynx`."""

    return {phrase: subgroup.name for subgroup in subgroup_list for phrase in subgroup.phrase_counter}


def is_relevant_subject(span: Span, subject_head_noun) -> bool:
    """Check if a span refers to the target subject or any of its subgroups."""

    return is_directly_relevant_subject(span, subject_head_noun) or is_coref_relevant_subject(span, subject_head_noun)


def is_directly_relevant_subject(span: Span, subject_head_noun) -> bool:
    """Check if a span directly contains the target subject."""

    return span.root.lemma_.lower() == subject_head_noun


def is_coref_relevant_subject(span: Span, subject_head_noun: str) -> bool:
    """Check if a span has co-reference that contains the target subject.
    This does not include the case that the span itself contains the target subject."""

    if is_directly_relevant_subject(span, subject_head_noun):
        return False

    if has_coref(span) and get_coref_main(span).root.lemma_.lower() == subject_head_noun:
        return True

    return False


def is_in_coref(token: Token) -> bool:
    """Check if a token is inside a span which belongs to a co-reference cluster."""

    # noinspection PyProtectedMember
    return token._.in_coref


def get_coref_clusters(token: Token) -> List:
    """Get co-reference clusters of a tokens."""

    if not is_in_coref(token):
        return []
    # noinspection PyProtectedMember
    return token._.coref_clusters


def has_coref(span: Span) -> bool:
    """Check if a span belongs to any co-reference cluster."""

    # noinspection PyProtectedMember
    return span._.is_coref


def get_coref_main(span: Span) -> Span:
    """Get the representative of co-reference cluster containing the given span."""

    # noinspection PyProtectedMember
    return span._.coref_cluster.main


def pre_check_assertion(assertion: Assertion) -> bool:
    """Set of rules to pre-check the validity of a relevant assertion."""

    # none object
    if assertion.obj is None:
        return False
    # comparative adjective object
    if is_comparative_adj(assertion.obj):
        return False
    # 1-word non-alphabetic object
    if len(assertion.full_obj) == 1 and not assertion.obj.is_alpha:
        return False
    # non-alphabetic predicate
    if not assertion.is_synthetic and not assertion.verb.is_alpha:
        return False
    # ignored pronouns occur in object
    if any([t.lower_ in PRONOUNS_IGNORED for t in assertion.full_obj]):
        return False
    # ignored pronouns occur in facets
    if any([t.lower_ in PRONOUNS_IGNORED for facet in assertion.facets if facet.full_statement is not None for t in
            facet.full_statement]):
        return False
    # 1-word pronoun object
    if len(assertion.full_obj) == 1 and assertion.full_obj[0].pos == symbols.PRON:
        return False
    # negative
    if not assertion.is_synthetic and any([t.dep == symbols.neg for t in assertion.full_pred]):
        return False

    return True


def get_nearest_before_subject_name(token: Token, subject: str, phrase2subgroup_name: Dict[str, str]) -> str:
    """Return the nearest subject/subgroup before the given token."""

    doc = token.doc
    candidates = []
    for noun_chunk in doc.noun_chunks:
        if noun_chunk[0].i >= token.i:
            break

        subject_name = normalize_subject_noun_chunk(noun_chunk)
        if subject_name == subject or subject_name in phrase2subgroup_name:
            candidates.append(subject_name)

    if not candidates:
        return ""

    return candidates[-1]


def extract_characteristics_of_subpart(full_obj: Span) -> List[Span]:
    """Extract all adjectives supporting the subpart which is the root word of the given span."""

    # root must be the ending word
    root = full_obj.root
    if root != full_obj[-1]:
        return []

    compound_indexes = [token.i for token in root.lefts if token.dep_ == "compound" and is_noun_or_proper_noun(token)]
    compound_indexes.append(root.i)
    i = min(compound_indexes)
    tokens = [token for token in full_obj
              if token.i < i
              and token.dep != symbols.det
              and (token.dep not in ADVERB_EDGES or is_special_adverb(token))
              and token.dep != symbols.poss
              and token.dep_ != "nummod"
              ]
    # filter
    if not tokens:
        return []
    if len(tokens) == 1 and not tokens[0].is_alpha:
        return []

    obj = get_span(tokens)

    # more filter
    if is_comparative_adj(obj.root):
        return []
    if not is_adjective(obj.root):
        return []

    # extract also conjuncts
    conjunct_object_list = [get_span([conjunct]) for conjunct in get_conjunctions(obj.root)]

    if not conjunct_object_list:
        return [obj]
    else:
        conjunct_object_list.append(get_span([obj.root]))
        return conjunct_object_list


def refers_to_subject_or_subgroups(token: Token, subject_head_noun: str) -> bool:
    """Check if a token is in a span which is relevant to the target subject through its co-reference cluster."""

    if is_in_coref(token) and get_coref_clusters(token)[0].main.root.lemma_ == subject_head_noun:
        return True

    return False


def filter_assertion_list(assertion_list: List[SimplifiedAssertion]) -> List[SimplifiedAssertion]:
    """Set of rules to post-process the assertion list."""

    result_list = []
    for assertion in assertion_list:
        if len(assertion.pred) <= 1:
            continue
        if len(str(assertion.obj)) <= 1:
            continue

        # # ignore PERSON
        # if any(token.ent_type == symbols.PERSON for token in assertion.obj):
        #     continue

        if len(assertion.obj) == 1:
            if assertion.obj.root.pos in ONE_WORD_OBJ_POS_IGNORED:
                continue
            if assertion.obj.root.like_num:
                continue
            if assertion.obj.root.lower_ in ALL_PRONOUNS.union(ONE_WORD_OBJECT_IGNORED):
                continue
            if str(assertion.obj).startswith("-"):
                continue
            # if assertion.obj.lower_.startswith("that "):
            #     continue

        is_ignored_predicate = False
        for prefix in IGNORED_PREDICATES:
            if assertion.pred.startswith(prefix):
                is_ignored_predicate = True
                break
        if is_ignored_predicate:
            continue

        if assertion.pred == 'be':
            if assertion.obj.root.lower_ == 'old':
                doc = assertion.obj.root.doc
                i = assertion.obj.root.i
                if doc[i - 1].lemma_ in {'day', 'month', 'year'}:
                    continue
            if assertion.obj.root.lower_ == assertion.subj.split()[-1]:
                continue
            if len(assertion.obj) >= 3 and (assertion.obj[0].lower_ + " " + assertion.obj[1].lower_) in {'the only',
                                                                                                         'the first'}:
                continue

        # limit predicate length
        if len(assertion.pred.split()) >= 4 or len(assertion.pred) == 0:
            continue

        if any([t.lemma_ == ',' for t in assertion.obj]):
            continue

        if assertion.obj.root.lemma_ == "ability":
            continue

        first_obj_token = assertion.obj[0]
        last_obj_token = assertion.obj[-1]

        if first_obj_token.lower_ == '’s' or first_obj_token.lower_ == "'s":
            continue
        if last_obj_token.lower_ == '’s' or last_obj_token.lower_ == "'s":
            continue

        if len(finalize_object(assertion.obj)) > MAX_OBJ_LEN:
            continue
        if len(finalize_object(assertion.obj, remove_all_punctuation=True, return_tokens=True)) == 0:
            continue

        if any([t.dep == symbols.neg for t in assertion.obj]):
            continue

        result_list.append(assertion)

    return result_list
