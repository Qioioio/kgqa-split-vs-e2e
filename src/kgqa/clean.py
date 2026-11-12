"""Cleaning LC-QuAD 2.0 and splitting it without leakage.

- Questions that are unusable (empty, "n/a", or still containing template braces) are replaced by
  their first paraphrase, or dropped if there is none.
- Queries that do not parse are dropped.
- A development set of 5% is cut from the training set by groups: questions with the same gold
  query or the same question text always stay on the same side.
- Test questions whose gold query or question text also occurs in the training set are dropped,
  and so are test questions whose gold query returned nothing or failed when the GRISP authors
  ran it (a first filter; the sampled questions are checked again on our own endpoint).

Every query is assigned a structure: the query with entities, properties, literals, numbers and
variable names replaced by placeholders. Structures are numbered by frequency in the raw
training set.
"""
import ast
import collections
import random
import re

SEED = 20261030
DEV_FRACTION = 0.05
# The five structures left out of the training data of the -B models (numbers as assigned by
# number_structures): two whose parts occur in other structures, and the three ORDER BY structures.
HELDOUT_STRUCTURES = {"comp": (4, 12), "zero": (24, 25, 26)}
PREFIX_DECL = re.compile(r"PREFIX\s+\w*:\s*<[^>]*>\s*", re.I)
NUMBER = re.compile(r"(?<![\w?:])[-+]?\d+(\.\d+)?([eE][-+]?\d+)?(?![\w])")
COUNT = re.compile(r"\bcount\s*\(", re.I)


def structure(sparql):
    """The query with everything question-specific replaced by placeholders."""
    s = PREFIX_DECL.sub("", sparql).replace("\n", " ")
    s = re.sub(r"'[^']*'", "'LIT'", s)
    s = re.sub(r'"[^"]*"', "'LIT'", s)
    s = re.sub(r"wd:[QP]\d+", "wd:E", s)
    s = re.sub(r"(wdt|p|ps|pq):P\d+", r"\1:R", s)
    s = NUMBER.sub("NUM", s)
    s = re.sub(r"\bt\d{6,}\b", "NUM", s)   # a few LC-QuAD queries compare with a bare token such as t1410874016
    names = []

    def rename(m):
        if m.group(0) not in names:
            names.append(m.group(0))
        return f"?v{names.index(m.group(0))}"

    s = re.sub(r"\?\w+", rename, s).lower()
    s = re.sub(r"\s*([{}().,;])\s*", r" \1 ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r" \. \}", " }", s)


def family(pattern):
    """The question type of a structure, decided in this order."""
    if pattern.startswith("ask"):
        return "ASK"
    if "count (" in pattern:
        return "COUNT"
    if "order by" in pattern:
        return "ORDER BY"
    if "rdfs:label" in pattern and re.search(r"\blimit\b", pattern):
        return "label filter + LIMIT"
    head = pattern.split(" where ")[0] if " where " in pattern else pattern.split(" { ")[0]
    if len(set(re.findall(r"\?v\d+", head))) >= 2:
        return "two columns"
    return "plain"


def number_structures(raw_train):
    """{structure: id}, 1 = the most frequent structure of the raw training set."""
    counts = collections.Counter(structure(x["sparql"]) for x in raw_train)
    return {s: i for i, (s, _) in enumerate(counts.most_common(), 1)}


def same_query_key(sparql):
    return re.sub(r"\s+", " ", PREFIX_DECL.sub("", sparql)).strip().lower()


def same_question_key(question):
    return re.sub(r"\s+", " ", question).strip().lower()


def _unusable(q):
    return q is None or not q.strip() or "{" in q or "}" in q or q.strip().lower() in ("n/a", "na")


def _first_paraphrase(item):
    p = item.get("paraphrases")
    if isinstance(p, str):
        s = p.strip()
        if s in ("", "[]"):
            return None
        try:
            p = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return s
    if isinstance(p, (list, tuple)):
        p = [str(t).strip() for t in p if t is not None and str(t).strip()]
        return p[0] if p else None
    return None if p is None else str(p).strip()


def choose_question(item):
    """(question text, "question" or "paraphrase"), or (None, None) if neither is usable."""
    q = item.get("question")
    q = q if isinstance(q, str) else None
    if not _unusable(q):
        return q.strip(), "question"
    p = _first_paraphrase(item)
    if not _unusable(p):
        return p, "paraphrase"
    return None, None


def syntax_error(sparql):
    """None if rdflib parses the query, else a short description of the error."""
    from rdflib.plugins.sparql.parser import parseQuery
    try:
        parseQuery(sparql)
        return None
    except Exception as e:   # any parser exception counts
        return f"{type(e).__name__}: {str(e)[:160]}"


def split_dev(records, fraction=DEV_FRACTION, seed=SEED):
    """(train, dev). Records that share a gold query or a question are kept together."""
    parent = list(range(len(records)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    seen_query, seen_question = {}, {}
    for i, r in enumerate(records):
        for key, seen in ((same_query_key(r["sparql"]), seen_query), (same_question_key(r["question"]), seen_question)):
            if key in seen:
                a, b = find(i), find(seen[key])
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                seen[key] = i
    groups = collections.defaultdict(list)
    for i in range(len(records)):
        groups[find(i)].append(i)
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    target, dev = round(fraction * len(records)), set()
    for root in order:
        if len(dev) >= target:
            break
        dev.update(groups[root])
    return [r for i, r in enumerate(records) if i not in dev], [r for i, r in enumerate(records) if i in dev]


def clean_lcquad(raw_train, raw_test, grisp_outputs, train_errors, test_errors):
    """-> (train, dev, test, removed). `grisp_outputs` are the per-question results published by the
    GRISP authors (id -> entry); only the size of their gold result is used. `*_errors` are the
    syntax_error() results for the raw records, in order."""
    ids = number_structures(raw_train)
    removed, kept = [], []
    for item, error in zip(raw_train, train_errors, strict=True):
        question, source = choose_question(item)
        if question is None:
            removed.append({"id": item["id"], "split": "train", "reason": "question_unusable"})
        elif error:
            removed.append({"id": item["id"], "split": "train", "reason": "sparql_parse_error", "detail": error})
        else:
            kept.append({"id": item["id"], "question": question, "question_source": source, "sparql": item["sparql"],
                         "structure": ids.get(structure(item["sparql"]), 0)})
    train, dev = split_dev(kept)

    train_queries = {same_query_key(x["sparql"]) for x in raw_train}   # the whole official training set
    train_questions = {same_question_key(r["question"]) for r in kept}
    test = []
    for item, error in zip(raw_test, test_errors, strict=True):
        question, source = choose_question(item)
        gold = ((grisp_outputs.get(item["id"]) or {}).get("target") or {})
        size = gold.get("size") or 0
        if question is None:
            removed.append({"id": item["id"], "split": "test", "reason": "question_unusable"})
        elif error:
            removed.append({"id": item["id"], "split": "test", "reason": "sparql_parse_error", "detail": error})
        elif same_query_key(item["sparql"]) in train_queries:
            removed.append({"id": item["id"], "split": "test", "reason": "dup_of_train", "detail": "gold"})
        elif same_question_key(question) in train_questions:
            removed.append({"id": item["id"], "split": "test", "reason": "dup_of_train", "detail": "question"})
        elif gold.get("err"):
            removed.append({"id": item["id"], "split": "test", "reason": "gold_error_grisp"})
        elif size == 0:
            removed.append({"id": item["id"], "split": "test", "reason": "gold_empty_grisp"})
        else:
            test.append({"id": item["id"], "question": question, "question_source": source, "sparql": item["sparql"],
                         "structure": ids.get(structure(item["sparql"]), 0), "gold_size_grisp": size,
                         "over_1024": size > 1024, "is_count": bool(COUNT.search(item["sparql"]))})
    return train, dev, test, removed
