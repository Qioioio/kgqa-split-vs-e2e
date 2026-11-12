"""Linking: turn a skeleton with names into a query with Wikidata IDs.

A skeleton writes entities as wd:<<name>> and properties as wdt:<<name>> (also p:, ps:, pq:).
- Entities are looked up with Wikidata's own entity search (wbsearchentities, top 10).
- Properties are looked up in a local table of all property labels and aliases:
  exact label match, then exact alias match, then nearest neighbours by embedding.
  Ties are ordered by how often the property occurs in the training queries.
The first candidate of every slot gives the query; further combinations, ordered by the
sum of candidate ranks, can be kept as retries.
"""
import collections
import copy
import heapq
import json
import os
import re
import time
import unicodedata

from .execute import USER_AGENT
from .files import read_jsonl
from .postprocess import extract_query
from .skeleton import SLOT, fill

PROPERTY_ID = re.compile(r"\b(?:wdt|p|ps|pq):(P\d+)\b")
TOP_K = 10
WDQS = "https://query.wikidata.org/sparql"


def normalize(name):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", name)).strip().lower()


# ---------------------------------------------------------------- entities
class EntitySearch:
    """wbsearchentities (items, English, top 10), called through the MWAPI service of the Wikidata
    Query Service so that one request can search a batch of names. Results are kept in a JSON-lines
    file; a name that is in the file is never searched again."""
    BATCH, PAUSE, EMPTY_RETRIES, EMPTY_PAUSE = 20, 1.0, 2, 5.0

    def __init__(self, cache_path):
        self.cache_path = cache_path
        self.cache = {r["key"]: r["ids"] for r in read_jsonl(cache_path)} if os.path.exists(cache_path) else {}
        self.session, self.last = None, 0.0

    def _wdqs(self, query, tries=8):
        import requests
        if self.session is None:
            self.session = requests.Session()
            self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"})
        wait = 5.0
        for _ in range(tries):
            gap = time.monotonic() - self.last
            if gap < self.PAUSE:
                time.sleep(self.PAUSE - gap)
            try:
                r = self.session.post(WDQS, data={"query": query}, timeout=(15, 180))
            except requests.exceptions.RequestException:
                self.last = time.monotonic()
                time.sleep(wait)
                wait = min(wait * 2, 300)
                continue
            self.last = time.monotonic()
            if r.status_code == 200:
                return r.json()["results"]["bindings"]
            if r.status_code in (429, 500, 502, 503, 504):
                retry_after = r.headers.get("Retry-After", "")
                time.sleep(min(float(retry_after), 300) if retry_after.isdigit() else wait)
                wait = min(wait * 2, 300)
                continue
            raise RuntimeError(f"WDQS {r.status_code}: {r.text[:200]}")
        raise RuntimeError("WDQS: giving up after retries")

    def _search_batch(self, names):
        values = " ".join('"' + n.replace("\\", "\\\\").replace('"', '\\"') + '"' for n in names)
        query = ("SELECT ?search ?item ?num WHERE { VALUES ?search { " + values + " }\n"
                 " SERVICE <http://wikiba.se/ontology#mwapi> {\n"
                 "  <http://www.bigdata.com/rdf#serviceParam> <http://wikiba.se/ontology#api> \"EntitySearch\" .\n"
                 "  <http://www.bigdata.com/rdf#serviceParam> <http://wikiba.se/ontology#endpoint> \"www.wikidata.org\" .\n"
                 "  <http://www.bigdata.com/rdf#serviceParam> <https://www.mediawiki.org/ontology#API/search> ?search .\n"
                 "  <http://www.bigdata.com/rdf#serviceParam> <https://www.mediawiki.org/ontology#API/language> \"en\" .\n"
                 "  ?item <http://wikiba.se/ontology#apiOutputItem> <https://www.mediawiki.org/ontology#API/item> .\n"
                 "  ?num <http://wikiba.se/ontology#apiOrdinal> true . } }")
        found = {n: [] for n in names}
        for b in self._wdqs(query):
            name, rank = b["search"]["value"], int(b["num"]["value"])
            if name in found and rank < TOP_K:
                found[name].append((rank, b["item"]["value"].rsplit("/", 1)[-1]))
        return {n: [i for _, i in sorted(pairs)] for n, pairs in found.items()}

    def lookup(self, name):
        """Cached candidates for `name`, or None if it has not been searched."""
        return self.cache.get(normalize(name))

    def search(self, names):
        """Search the names that are not cached yet (in batches) and add them to the cache file."""
        todo = list(dict.fromkeys(n for n in names if n.strip() and self.lookup(n) is None))
        for k in range(0, len(todo), self.BATCH):
            batch = todo[k:k + self.BATCH]
            found = self._search_batch(batch)
            # the service now and then returns nothing for some names of a batch without an error,
            # so empty results are searched again before they are accepted
            for _ in range(self.EMPTY_RETRIES):
                empty = [n for n in batch if not found[n]]
                if not empty:
                    break
                time.sleep(self.EMPTY_PAUSE)
                found.update(self._search_batch(empty))
            stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with open(self.cache_path, "a", encoding="utf-8") as f:
                for name, ids in found.items():
                    self.cache[normalize(name)] = ids
                    f.write(json.dumps({"key": normalize(name), "name": name, "ids": ids, "ts": stamp}, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- properties
def property_frequencies(train_file):
    """In how many training queries each property occurs (the file holds chat messages whose
    assistant turn is the full query)."""
    counts = collections.Counter()
    for row in read_jsonl(train_file):
        target = next(m["content"] for m in row["messages"] if m["role"] == "assistant")
        counts.update(set(PROPERTY_ID.findall(target)))
    return counts


class PropertyIndex:
    def __init__(self, labels_path, frequencies, embedder=None, vector_cache=None):
        self.frequencies = frequencies
        self.by_label, self.by_alias = collections.defaultdict(set), collections.defaultdict(set)
        self.strings = []   # (label or alias, property id), in file order
        for r in read_jsonl(labels_path):
            if r.get("missing") or not r.get("label"):
                continue
            self.by_label[normalize(r["label"])].add(r["id"])
            self.strings.append((r["label"], r["id"]))
            for alias in r.get("aliases") or []:
                self.by_alias[normalize(alias)].add(r["id"])
                self.strings.append((alias, r["id"]))
        self.embedder = self.vectors = None
        if embedder:
            import numpy as np
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(embedder)
            if vector_cache and os.path.exists(vector_cache):
                self.vectors = np.load(vector_cache)
            else:
                self.vectors = self.embedder.encode([s for s, _ in self.strings], convert_to_numpy=True,
                                                    normalize_embeddings=True, batch_size=256)
                if vector_cache:
                    np.save(vector_cache, self.vectors)
            if len(self.vectors) != len(self.strings):
                raise ValueError("the vector cache does not belong to this label file")

    def with_frequencies(self, frequencies):
        """The same index (tables and vectors shared) with another frequency table for ordering ties."""
        other = copy.copy(self)
        other.frequencies = frequencies
        return other

    def _ordered(self, ids):
        return sorted(ids, key=lambda p: (-self.frequencies.get(p, 0), int(p[1:])))

    def candidates(self, name, k=TOP_K):
        """Up to k property ids for a name, best first."""
        key, out = normalize(name), []
        for table in (self.by_label, self.by_alias):
            out += [p for p in self._ordered(table.get(key, ())) if p not in out]
        if len(out) < k and self.vectors is not None:
            q = self.embedder.encode([name], convert_to_numpy=True, normalize_embeddings=True)[0]
            scores = self.vectors @ q
            for i in scores.argsort()[::-1][: k * 20]:
                p = self.strings[i][1]
                if p not in out:
                    out.append(p)
                if len(out) >= k:
                    break
        return out[:k]


# ---------------------------------------------------------------- skeleton -> query
def named_slots(query):
    """[(start, end, prefix, name)] in text order."""
    return [(m.start(), m.end(), m.group(1), m.group(2)) for m in SLOT.finditer(query)]


def rank_combinations(n_candidates, limit):
    """The first `limit` combinations of candidate ranks, ordered by the sum of ranks; among equal
    sums, a later candidate for an earlier slot comes first. Starts with all-first."""
    if not n_candidates or any(n == 0 for n in n_candidates):
        return []
    key = lambda ranks: (sum(ranks), tuple(-x for x in ranks))  # noqa: E731
    start = tuple(0 for _ in n_candidates)
    heap, seen, out = [(key(start), start)], {start}, []
    while heap and len(out) < limit:
        _, ranks = heapq.heappop(heap)
        out.append(ranks)
        for i in range(len(ranks)):
            if ranks[i] + 1 < n_candidates[i]:
                nxt = ranks[:i] + (ranks[i] + 1,) + ranks[i + 1:]
                if nxt not in seen:
                    seen.add(nxt)
                    heapq.heappush(heap, (key(nxt), nxt))
    return out


def link(output, entity_candidates, property_candidates, retries=0):
    """Model output -> {"query", "slots", "retry_queries"}.

    `query` is None when no query can be extracted from the output or when a slot has no
    candidate. The two lookup functions map a name to a ranked list of ids."""
    query, _ = extract_query(output)
    if query is None:
        return {"query": None, "slots": [], "retry_queries": []}
    slots = named_slots(query)
    cands = [(entity_candidates if prefix == "wd" else property_candidates)(name)[:TOP_K] for _, _, prefix, name in slots]
    out = {"query": None,
           "slots": [{"prefix": prefix, "name": name, "candidates": c} for (_, _, prefix, name), c in zip(slots, cands, strict=True)],
           "retry_queries": []}
    if not slots:
        out["query"] = query
    elif all(cands):
        combos = rank_combinations([len(c) for c in cands], 1 + retries)
        queries = [fill(query, [c[r] for c, r in zip(cands, ranks, strict=True)]) for ranks in combos]
        out["query"], out["retry_queries"] = queries[0], queries[1:]
    return out
