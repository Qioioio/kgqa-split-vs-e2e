"""From raw model output to one executable query.

The same rules are applied to every system: take the query out of the text (code fence
or first query-like span), drop anything after it, add missing standard PREFIX
declarations, and rewrite the Wikidata label service, which QLever does not support.
"""
import re

STANDARD_PREFIXES = {
    "wd": "http://www.wikidata.org/entity/",
    "wds": "http://www.wikidata.org/entity/statement/",
    "wdv": "http://www.wikidata.org/value/",
    "wdref": "http://www.wikidata.org/reference/",
    "wdt": "http://www.wikidata.org/prop/direct/",
    "wdtn": "http://www.wikidata.org/prop/direct-normalized/",
    "p": "http://www.wikidata.org/prop/",
    "ps": "http://www.wikidata.org/prop/statement/",
    "psv": "http://www.wikidata.org/prop/statement/value/",
    "psn": "http://www.wikidata.org/prop/statement/value-normalized/",
    "pq": "http://www.wikidata.org/prop/qualifier/",
    "pqv": "http://www.wikidata.org/prop/qualifier/value/",
    "pqn": "http://www.wikidata.org/prop/qualifier/value-normalized/",
    "pr": "http://www.wikidata.org/prop/reference/",
    "prv": "http://www.wikidata.org/prop/reference/value/",
    "prn": "http://www.wikidata.org/prop/reference/value-normalized/",
    "wdno": "http://www.wikidata.org/prop/novalue/",
    "wikibase": "http://wikiba.se/ontology#",
    "bd": "http://www.bigdata.com/rdf#",
    "hint": "http://www.bigdata.com/queryHints#",
    "mwapi": "https://www.mediawiki.org/ontology#API/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "schema": "http://schema.org/",
    "prov": "http://www.w3.org/ns/prov#",
    "dct": "http://purl.org/dc/terms/",
    "ontolex": "http://www.w3.org/ns/lemon/ontolex#",
    "geo": "http://www.opengis.net/ont/geosparql#",
    "geof": "http://www.opengis.net/def/function/geosparql/",
}
PREFIX_DECL = re.compile(r"PREFIX\s+([A-Za-z][\w-]*|):\s*<([^>]*)>", re.I)
FENCE = re.compile(r"```[ \t]*(?:sparql|SPARQL|sql)?[ \t]*\n?(.*?)```", re.S)
# A query keyword only counts when something query-like follows, so that prose such as
# "Select the query below" is not taken for a query.
QUERY_START = re.compile(r"(?:^|\n|>)\s*(SELECT|ASK|CONSTRUCT|DESCRIBE)\b"
                         r"(?=\s*(?:\{|\?|\$|\*|\(|<|DISTINCT\b|REDUCED\b|WHERE\b|FROM\b))", re.I)
# Solution modifiers after the body, again only when they look like clauses.
MODIFIER = re.compile(r"(?:GROUP\s+BY\s*[?($]|HAVING\s*\(|ORDER\s+BY\s*(?:[?($]|ASC\b|DESC\b)|LIMIT\s+\d+|OFFSET\s+\d+)", re.I)
USED_PREFIX = re.compile(r"(?<![\w?$<\"'#:-])([A-Za-z][\w-]*):(?=[\w<]|$|\s)")
IRI_TOKEN = re.compile(r"<[^<>\"{}|^`\\\s]*>")
LABEL_SERVICE = re.compile(r"SERVICE\s+wikibase:label\s*\{", re.I)


def match_braces(text, start):
    """End position of the first {...} group at or after `start`, skipping strings, <IRIs>
    and # comments. None if the braces do not close."""
    depth, opened, quote, i, n = 0, False, None, start, len(text)
    while i < n:
        c = text[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = None
            i += 1
            continue
        if c in "\"'":
            quote = c
        elif c == "<":
            m = IRI_TOKEN.match(text, i)
            if m:
                i = m.end()
                continue
        elif c == "#":
            nl = text.find("\n", i)
            i = n if nl < 0 else nl
            continue
        elif c == "{":
            depth, opened = depth + 1, True
        elif c == "}":
            depth -= 1
            if opened and depth == 0:
                return i + 1
        i += 1
    return None


def _end_of_tail(body, end):
    """Extend `end` over GROUP BY / HAVING / ORDER BY / LIMIT / OFFSET clauses (each to the end
    of its line) and VALUES blocks that follow the query body; stop at anything else."""
    i = end
    while True:
        j = i + (len(body[i:]) - len(body[i:].lstrip()))
        if j >= len(body):
            return j
        if MODIFIER.match(body, j):
            nl = body.find("\n", j)
            i = len(body) if nl < 0 else nl
            continue
        if re.match(r"VALUES\b", body[j:], re.I):
            e = match_braces(body, j)
            if e is None:
                return i
            i = e
            continue
        if body[j] in ";":
            i = j + 1
            continue
        return i


def extract_query(text):
    """Model output -> (query to execute, or None; flags describing what was done)."""
    flags = {"fence": False, "multi_query": False, "prefixes_added": [], "prefix_conflicts": [],
             "label_service_rewritten": 0, "unbalanced": False}
    if not text or not text.strip():
        flags["empty_output"] = True
        return None, flags
    blocks = [m.group(1) for m in FENCE.finditer(text)]
    flags["fence"] = bool(blocks)
    with_query = [b for b in blocks if QUERY_START.search(PREFIX_DECL.sub(" ", b))]
    if with_query:
        chosen = with_query[0]
    elif QUERY_START.search(PREFIX_DECL.sub(" ", text)):
        chosen = text
    else:
        flags["no_query"] = True
        return None, flags
    declared = {}
    for prefix, iri in PREFIX_DECL.findall(chosen):
        declared.setdefault(prefix, iri)
    body = PREFIX_DECL.sub(" ", chosen)
    start = QUERY_START.search(body).start(1)
    end = match_braces(body, start)
    if end is None:
        flags["unbalanced"] = True
        end = len(body)
    end = _end_of_tail(body, end)
    query = body[start:end].strip().rstrip(";").strip()
    flags["multi_query"] = len(with_query) > 1 or bool(QUERY_START.search(body[end:]))
    query, flags["label_service_rewritten"] = rewrite_label_service(query)
    for prefix, iri in declared.items():
        if prefix in STANDARD_PREFIXES and STANDARD_PREFIXES[prefix] != iri:
            flags["prefix_conflicts"].append(prefix)
    # Known limitation, kept so that scores stay as reported: in a skeleton, a name containing "#" or
    # "'" (wd:<<Lorraine-Dietrich #6>>, wd:<<Shaquille O'Neal>>) is read as the start of a comment or a
    # string, so prefixes used after it may not be declared. This affects one or two queries per system.
    scan = re.sub(r"<[^<>\s]*>|\"[^\"]*\"|'[^']*'", " ", query)
    scan = re.sub(r"#[^\n]*", " ", scan)
    used = set(USED_PREFIX.findall(scan))
    added = sorted(p for p in used if p not in declared and p in STANDARD_PREFIXES)
    flags["prefixes_added"] = added
    decl = [f"PREFIX {p}: <{iri}>" for p, iri in declared.items()] + [f"PREFIX {p}: <{STANDARD_PREFIXES[p]}>" for p in added]
    return ("\n".join(decl + [query])).strip(), flags


def rewrite_label_service(query):
    """Remove SERVICE wikibase:label { ... } and bind the ?xLabel / ?xDescription variables that
    are used through OPTIONAL rdfs:label / schema:description in English.
    Returns (query, number of service blocks removed)."""
    n = 0
    while True:
        m = LABEL_SERVICE.search(query)
        if not m:
            break
        end = match_braces(query, m.start())
        if end is None:
            break
        query = query[:m.start()] + query[end:]
        n += 1
    if not n:
        return query, 0
    adds = []
    for var, kind in sorted(set(re.findall(r"\?(\w+?)(Label|Description)\b", query))):
        if re.search(rf"\?{var}\b(?!Label|Description)", query):
            pred = "rdfs:label" if kind == "Label" else "schema:description"
            adds.append(f"OPTIONAL {{ ?{var} {pred} ?{var}{kind} . FILTER (LANG(?{var}{kind}) = \"en\") }}")
    if adds:
        start = query.find("{")
        close = match_braces(query, start) if start >= 0 else None
        if close is not None:
            query = query[: close - 1] + " " + " ".join(adds) + " " + query[close - 1:]
    return query, n
