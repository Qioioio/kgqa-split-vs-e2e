"""Query skeletons: the gold query with every Wikidata ID replaced by a name.

    wd:Q1045 p:P1082 ?s    ->    wd:<<Somalia>> p:<<population>> ?s

Entities (wd:) and properties (wdt:, p:, ps:, pq:) become prefix:<<label>>, with the English
label (or the language-neutral "mul" label if there is no English one). Everything else in the
query is left as it is. A query with an ID that has no label cannot be turned into a skeleton.
"""
import re

from .clean import PREFIX_DECL

ID = re.compile(r"\b(wd|wdt|p|ps|pq):([QP]\d+)\b")
SLOT = re.compile(r"\b(wd|wdt|p|ps|pq):<<(.*?)>>")


def strip_prefixes(sparql):
    return PREFIX_DECL.sub("", sparql).strip()


def to_skeleton(sparql, labels):
    """-> (skeleton, slots) with slots = [(prefix, id, label language)] in text order, or
    (None, ids without a label). `labels` maps an ID to {"label", "label_lang"}."""
    slots, missing = [], []

    def name(m):
        prefix, wid = m.groups()
        record = labels.get(wid)
        if not record or not record.get("label"):
            missing.append(wid)
            return m.group(0)
        slots.append((prefix, wid, record["label_lang"]))
        return f"{prefix}:<<" + record["label"].replace("<<", "< <").replace(">>", "> >") + ">>"

    skeleton = ID.sub(name, strip_prefixes(sparql))
    return (None, missing) if missing else (skeleton, slots)


def fill(skeleton, ids):
    """Put IDs back into the slots of a skeleton, in order."""
    it = iter(ids)
    return SLOT.sub(lambda m: f"{m.group(1)}:{next(it)}", skeleton)


ANY_SLOT = re.compile(r"\b(wd|wdt|p|ps|pq):(?:([QP]\d+)\b|<<(.*?)>>)")


def find_slots(query):
    """IDs and names in text order: [(start, end, prefix, kind, id or None, name or None)], with
    kind "E" for entities (wd:) and "P" for properties."""
    return [(m.start(), m.end(), m.group(1), "E" if m.group(1) == "wd" else "P", m.group(2), m.group(3))
            for m in ANY_SLOT.finditer(query or "")]


def ids_by_kind(sparql):
    """{"E": [entity ids], "P": [property ids]} in text order, repeats included."""
    out = {"E": [], "P": []}
    for *_, kind, wid, _name in find_slots(sparql):
        if wid is not None:
            out[kind].append(wid)
    return out


def label_names(label_rows):
    """ID -> the set of its label and aliases, case-folded."""
    out = {}
    for r in label_rows:
        names = {(a or "").strip().casefold() for a in r.get("aliases") or []}
        if r.get("label"):
            names |= {r["label"].strip().casefold(), r["label"].replace("<<", "< <").replace(">>", "> >").strip().casefold()}
        out[r["id"]] = names - {""}
    return out
