"""Contains a lot of supporting functions towards assertion extraction."""

from typing import List, Optional, Set

from nltk.corpus import wordnet as wn
from spacy import symbols
from spacy.matcher import Matcher
from spacy.tokens.span import Span
from spacy.tokens.token import Token

from helper.constants import PREPOSITIONS, NUMERAL_ADJECTIVES, SPECIAL_PHRASE_PATTERNS

# to find object
EDGES_OF_OBJECTS = {symbols.dobj, symbols.nmod, symbols.oprd, symbols.acomp}  # , symbols.advcl, symbols.ccomp

# to complete predicates
EDGES_ALLOWED_IN_PREDICATES = {"xcomp", "auxpass", "aux", "compound", "neg", "prt"}  # advmod

# to complete noun phrases
EDGES_ALLOWED_IN_PHRASES = {"det", "amod", "compound", "nummod"}  # advmod

PREPOSITIONS_ALLOWED_IN_PHRASES = {"of", "than", "among", "amongst", "per"}
COMPARATIVE_SUPPORTING_WORDS = {"most", "more", "less", "least"}
ADVERBS_ALLOWED_IN_PHRASES = {"around", "about", "mass"}.union(COMPARATIVE_SUPPORTING_WORDS)

# prepositions
PREPOSITION_EDGES = {symbols.prep, symbols.agent}

# adverb
ADVERB_EDGES = {symbols.advmod, symbols.npadvmod}

# conjunctive
CONJUNCTIVE_WORDS = {"and", "or"}

# synthetic verbs
TOBE = "$be$"  # DO NOT CHANGE THIS

# subgroup/subpart phrase
EDGES_IGNORED_IN_SUBGROUP_PHRASES = {"det", "nummod", "advmod", "punct", "preconj", "poss"}


def find_object(verb: Token) -> List[Token]:
    """Extract objects of the given verb. Essentially StuffIE's original algorithm."""

    object_list = []

    # be + adj/noun
    # different parsing made by SpaCY
    if verb.lemma_.lower() == "be":
        for child in verb.children:
            if child.dep in {symbols.acomp, symbols.attr}:
                object_list.append(child)
                object_list.extend(get_conjunctions(child))  # also extract conjuncts
                return object_list

    # original Stuffie algorithm
    for child in verb.children:
        if child.dep in EDGES_OF_OBJECTS:
            object_list.append(child)
            object_list.extend(get_conjunctions(child))  # also extract conjuncts
            return object_list

    # "lions have been known to hunt humans"
    # <lions; have been known to hunt; humans>
    for child in verb.children:
        if child.dep == symbols.xcomp and child.pos == symbols.VERB:
            return find_object(child)

    # conjuncts
    # ex: "An inventor creates or discovers a new method and device."
    # <an inventor; creates; a new method>
    # <an inventor; creates; device>
    # <an inventor; discovers; a new method>
    # <an inventor; discovers; device>
    conj_verb = get_target(verb, symbols.conj)
    flag = any(child.lower_ in CONJUNCTIVE_WORDS for child in verb.children)
    if flag and conj_verb is not None and conj_verb.pos == symbols.VERB \
            and get_target(verb, symbols.prep) is None and get_target(verb, symbols.agent) is None:
        return find_object(conj_verb)

    return []


def get_span(token_list) -> Optional[Span]:
    """Sort the tokens by their index, then return the span from the first until the last token."""

    if not token_list:
        return token_list

    tokens = sorted(token_list, key=lambda token: token.i)
    start = tokens[0].i
    end = tokens[-1].i + 1
    phrase = tokens[0].doc[start:end]

    # complete left-right punctuation
    num_left_punct = sum([token.is_left_punct for token in phrase])
    num_right_punct = sum([token.is_right_punct for token in phrase])
    if num_left_punct == num_right_punct + 1:
        for token in phrase.doc[end:]:
            if token.i > phrase.sent[-1].i:
                break
            if token.is_right_punct:
                phrase = phrase.doc[start:(token.i + 1)]
                break

    # complete quotation marks
    num_quotation_marks = sum([token.is_quote for token in phrase])
    if num_quotation_marks % 2 == 1:
        for token in phrase.doc[end:]:
            if token.i > phrase.sent[-1].i:
                break
            if token.is_quote:
                phrase = phrase.doc[start:(token.i + 1)]
                break

    return phrase


def find_long_phrase(head_word: Token, prep_in: Set = None, prep_not_in: Set = None) -> Optional[Span]:
    """This is the main technique to complete subject/object.
    The main difference between this and the `find_short_noun_phrase` function
    is that this allows noun phrases to have prepositions inside them."""

    if prep_not_in is None:
        prep_not_in = set()

    if prep_in is None:
        prep_in = PREPOSITIONS_ALLOWED_IN_PHRASES

    if head_word is None:
        return None

    # no preposition for adjective phrase
    if head_word.pos_ == "ADJ" and not is_comparative_adj(head_word):
        prep_in = []
        prep_not_in = PREPOSITIONS
    # all preposition for infinite verb
    elif head_word.tag_ == "VB":
        prep_in = []
        prep_not_in = []

    # special phrase matcher
    special_phrase_matcher = Matcher(head_word.vocab)
    for key, patterns in SPECIAL_PHRASE_PATTERNS.items():
        special_phrase_matcher.add(key, None, patterns)

    matches = special_phrase_matcher(head_word.doc)
    for _, start, end in matches:
        span = head_word.doc[start:end]
        if span.root == head_word:
            return span

    # other cases
    return get_span(recursive_find_long_phrase(head_word, prep_in, prep_not_in))


def recursive_find_long_phrase(head_word: Token, prep_in: Set[str], prep_not_in: Set[str]) -> List[Token]:
    """Start with short phrase, then extend it to include allowed prepositions and their direct objects"""

    short_phrase = find_short_phrase(head_word)
    result_token_list = [token for token in short_phrase]

    for child in head_word.children:
        if not is_valid_preposition_node(child, prep_in, prep_not_in):
            continue
        for grandchild in child.children:
            if grandchild.dep == symbols.pobj:  # direct object exists
                result_token_list.extend(recursive_find_long_phrase(grandchild, prep_in, prep_not_in))
                for conj in grandchild.conjuncts:  # also extract conjuncts
                    result_token_list.extend(recursive_find_long_phrase(conj, prep_in, prep_not_in))

    return result_token_list


def is_valid_preposition_node(node: Token, prep_in: Set[str], prep_not_in: Set[str]) -> bool:
    """Check validation of a node being preposition."""

    return node.dep in PREPOSITION_EDGES and (node.lower_ in prep_in or len(prep_in) == 0) and (
            node.lower_ not in prep_not_in)


def find_short_phrase(head_word: Token, uses_exact_spacy_noun_chunk: bool = False) -> Optional[Span]:
    """This is based on scanning for a limit set of dependency edges.
    This is essentially motivated by StuffIE's original algorithm,
    but also relies on the `noun_chunks` function of SpaCy.
    See also `EDGES_ALLOWED_IN_NOUN_PHRASES`."""

    if head_word is None:
        return None

    token_list = []

    # SpaCy's noun chunk function comes first
    for noun_chunk in head_word.sent.noun_chunks:
        if noun_chunk.root != head_word:
            continue
        if uses_exact_spacy_noun_chunk:
            return noun_chunk
        else:
            token_list = [token for token in noun_chunk if (token.lower_ in ADVERBS_ALLOWED_IN_PHRASES) or not (
                    token.dep in ADVERB_EDGES and token.head == noun_chunk.root) or is_special_adverb(token)]
            break

    # StuffIE's original algorithm comes to the rescue
    if len(token_list) == 0:
        token_list = [head_word]
        for child in head_word.children:
            if child.dep_ in EDGES_ALLOWED_IN_PHRASES or (
                    child.lower_ in ADVERBS_ALLOWED_IN_PHRASES) or is_special_adverb(child):
                token_list.append(child)

    # right-trim adverbs
    token_list = sorted(token_list, key=lambda x: x.i)
    if len(token_list) > 1 and token_list[-1].lower_ in ADVERBS_ALLOWED_IN_PHRASES:
        token_list.pop()

    return get_span(token_list)


def is_special_adverb(token: Token) -> bool:
    if token is None:
        return False

    return token.dep in ADVERB_EDGES and (token.lower_.endswith("-") or (
            token.i < len(token.doc) - 1 and token.doc[token.i + 1].lower_.startswith("-")))


def find_head_of_preposition_facet(prep: Token) -> Optional[Token]:
    """Given a preposition, extract its direct object through the `pobj` edge."""

    pobj_children = [node for node in prep.children if node.dep == symbols.pobj]
    if len(pobj_children) > 0:
        return pobj_children[0]

    return None


def get_conjunctions(node: Token) -> List[Token]:
    """Get all conjuncts of a token if they are connected by `and` or `or`."""

    conjunct_list = node.conjuncts

    has_allowed_conjunctive_words = (
            any([any([child.lower_ in CONJUNCTIVE_WORDS for child in conjunct.children]) for conjunct in conjunct_list])
            or any([child.lower_ in CONJUNCTIVE_WORDS for child in node.children])
    )

    return list(conjunct_list) if has_allowed_conjunctive_words else []


def complete_predicate(head_verb: Token, obj: Token = None) -> List[Token]:
    """The StuffIE algorithm to extract full predicate given the head verb."""

    pred = [head_verb]
    for child in head_verb.children:
        if child.dep_ in EDGES_ALLOWED_IN_PREDICATES:
            if obj is not None and child.i > obj.i:
                continue
            if child.pos == symbols.VERB:
                pred.extend(complete_predicate(child))
            else:
                pred.append(child)
    pred = sorted(pred, key=lambda t: t.i)

    return pred


def get_target(source: Token, edge) -> Optional[Token]:
    """Get the target node of an edge given the source node and the edge's name."""

    for child in source.children:
        if child.dep == edge:
            return child
    return None


def chunk_ends_with_tokens(chunk: Span, token_list: List[str]) -> bool:
    """Check if a span ends with a list of words. Words are lemmatized."""

    if len(token_list) > len(chunk):
        return False

    for i in range(1, len(token_list) + 1):
        j = -i
        if token_list[j] != chunk[j].lemma_.lower():
            return False

    return True


def find_compound_noun(head_noun: Token) -> str:
    """Extract the compound noun around the given head noun.
    Tokens added are the ones occurring in the left of the head noun and connected to it by the `compound` edge.
    Mostly exclusively used for associated terms extraction."""

    arr = [head_noun]
    arr.extend([token for token in head_noun.lefts if token.dep_ == "compound" and is_noun_or_proper_noun(token)])
    arr = sorted(arr, key=lambda x: x.i)
    return " ".join(find_lemma_of_noun(token) for token in arr)


def find_lemma_of_noun(noun: Token) -> str:
    """Apply both the `morphy` function from `wordnet` and the `lemma_` property from SpaCy."""

    wordnet_lemma = wn.morphy(noun.lower_, wn.NOUN)
    spacy_lemma = noun.lemma_.lower()

    if wordnet_lemma is not None and len(wordnet_lemma) <= len(spacy_lemma):
        return wordnet_lemma

    return spacy_lemma


def remove_redundancy_from_subgroup_chunk(chunk: Span) -> List[Token]:
    """Perform normalization on a candidate subgroup noun chunk."""

    # get rid of unrelated dependency edges
    simplified_chunk = set([token for token in chunk if token.dep_ not in EDGES_IGNORED_IN_SUBGROUP_PHRASES])

    # get rid of stop words and punctuations
    simplified_chunk = set([token for token in simplified_chunk if not token.is_stop and not token.is_punct])

    # get rid of comparative adjectives
    simplified_chunk = set([token for token in simplified_chunk if not is_comparative_adj(token)])

    # all tokens should be children of root noun
    simplified_chunk = [token for token in simplified_chunk if token == chunk.root or token.head in simplified_chunk]

    # sort tokens by index
    simplified_chunk = sorted(simplified_chunk, key=lambda n: n.i)

    return simplified_chunk


def lemmatize_token_list(token_list: List[Token]) -> str:
    """Lemmatize all nouns in the list, then concatenate all tokens into a lower-cased string."""

    return " ".join(
        (token.lower_ if not is_noun_or_proper_noun(token) else find_lemma_of_noun(token)) for token in token_list)


def normalize_subject_noun_chunk(chunk: Span) -> str:
    """Given a noun chunk which can be subject of an assertion,
    return a normalized string after removing redundancy from the chunk."""

    return lemmatize_token_list(remove_redundancy_from_subgroup_chunk(chunk))


def is_adjective(word: Token) -> bool:
    """Check if a token being an adjective. Extended to also cover past participles."""

    return any([
        word.pos_ == 'ADJ',
        word.tag_ == 'JJ',
        word.tag_ == 'VBN',
        word.tag_ == 'VBD',
    ])


def is_comparative_adj(word: Token) -> bool:
    """Check if a token being a comparative or superlative adjective, using POS tags."""

    if not word.pos_ == "ADJ":
        return False

    if len(set([child.lower_ for child in word.children]).intersection(COMPARATIVE_SUPPORTING_WORDS)) > 0:
        return True

    return word.tag_ == "JJR" or word.tag_ == "JJS"


def is_noun_or_proper_noun(word: Token) -> bool:
    """This is necessary as SpaCy is usually mistaken between nouns and proper nouns."""

    return word.pos == symbols.NOUN or word.pos == symbols.PROPN


def is_modifier_of_superlative(token):
    """Check if a token being a support word to a superlative adjective."""

    return token.lower_ in NUMERAL_ADJECTIVES or token.lower_ in COMPARATIVE_SUPPORTING_WORDS


def has_superlative_or_distinctive(obj: Span):
    for token in obj:
        if token.lower_ in {"most", "least", "only"}:
            return True
        if token.tag_ == "JJS":
            return True
    return False


def finalize_object(obj: Span, remove_all_punctuation: bool = False, return_tokens: bool = False):
    tokens = [t for t in obj]

    # trailing punctuation 
    while len(tokens) > 0 and tokens[0].is_left_punct:
        tokens.pop(0)
    while len(tokens) > 0 and tokens[-1].is_right_punct:
        tokens.pop()

    # inside quotes
    tokens = [t for t in tokens if not t.is_left_punct and not t.is_right_punct]

    # adverbs (except for "more", "most", "second <largest>",...)
    tokens = [t for t in tokens if not (t.pos == symbols.ADV and not is_modifier_of_superlative(
        t) and t.head == obj.root and not is_special_adverb(t))]

    # determinants
    if not has_superlative_or_distinctive(obj):
        head_noun = obj.root
        adjectives = [token for token in tokens if token in head_noun.lefts and token.dep == symbols.amod]
        if len(adjectives) > 0:
            if tokens[0].dep == symbols.det:
                tokens.pop(0)
            adverbs = [token for adj in adjectives for token in adj.lefts if
                       token in tokens and token.pos == symbols.ADV and not is_special_adverb(token)]
            tokens = [token for token in tokens if token not in adverbs]

    if len(tokens) == 2 and tokens[0].dep == symbols.det and is_noun_or_proper_noun(tokens[1]) \
            and tokens[1].lower_ != "lot":
        head_noun = find_lemma_of_noun(tokens[1])
        if head_noun != tokens[1].lower_ or tokens[0].lower_ in {"a", "an"}:
            tokens.pop(0)

    # remove punctuations if requested
    if remove_all_punctuation:
        tokens = [t for t in tokens if not t.is_punct]

    # return tokens if requested
    if return_tokens:
        return [t.lower_ for t in tokens]

    # lemmatize if only one-word noun
    if len(tokens) == 1 and is_noun_or_proper_noun(tokens[0]):
        return find_lemma_of_noun(tokens[0])

    return ''.join([t.text_with_ws for t in tokens]).strip().lower()
