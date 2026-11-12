"""Per-question scoring.

A prediction is scored by comparing the result of its query with the result of
the gold query, both executed on the same endpoint. The score is the answer-set
F1 of GRASP (grasp_metrics.py), computed after normalising literals.
"""
import decimal
import re

from .clean import PREFIX_DECL
from .grasp_metrics import AskResult, SelectResult, f1_score

EXACT_AFTER = 1024      # GRASP's default: larger results are compared row by row instead of by assignment
MAX_GOLD_ROWS = 1024    # gold results larger than this are not used
XSD = "http://www.w3.org/2001/XMLSchema#"
NUMERIC_TYPES = {XSD + t for t in (
    "integer", "int", "long", "short", "byte", "decimal", "double", "float", "nonNegativeInteger",
    "positiveInteger", "negativeInteger", "nonPositiveInteger", "unsignedInt", "unsignedLong",
    "unsignedShort", "unsignedByte")}
DATE_TYPES = {XSD + "dateTime", XSD + "date"}
MIDNIGHT = re.compile(r"^(-?\d{4,})-(\d{2})-(\d{2})(?:T00:00:00(?:\.0+)?(?:Z|[+-]00:00)?)?$")
TOP_LEVEL_COUNT = re.compile(r"^\s*select\s+(?:distinct\s+)?\(\s*count\s*\(", re.I)


def normalize_term(term):
    """Make equal values compare equal: numbers by value, midnight timestamps as dates,
    strings without language tag or xsd:string. IRIs and blank nodes are left alone."""
    typ = term.get("type")
    if typ == "typed-literal":
        typ = "literal"
    if typ != "literal":
        return {"type": typ, "value": term.get("value")}
    value, datatype, lang = term.get("value", ""), term.get("datatype"), term.get("xml:lang")
    if datatype in NUMERIC_TYPES:
        try:
            x = decimal.Decimal(value.strip())
            if x.is_finite():
                s = format(x.normalize(), "f")
                return {"type": "literal", "value": "0" if s in ("-0", "0") else s, "datatype": "urn:x-numeric"}
        except decimal.InvalidOperation:
            pass
        return {"type": "literal", "value": value, "datatype": datatype}
    if datatype in DATE_TYPES:
        m = MIDNIGHT.match(value.strip())
        if m:
            return {"type": "literal", "value": "-".join(m.groups()), "datatype": "urn:x-date"}
        return {"type": "literal", "value": value.strip(), "datatype": datatype}
    if datatype is None or datatype == XSD + "string" or lang:
        return {"type": "literal", "value": value}
    return {"type": "literal", "value": value, "datatype": datatype}


def to_grasp(result):
    """A stored execution result (see execute.py) as a GRASP AskResult / SelectResult."""
    if "boolean" in result:
        return AskResult(bool(result["boolean"]))
    rows = [{var: normalize_term(term) for var, term in (row or {}).items()} for row in result.get("rows", [])]
    return SelectResult(list(result.get("vars", [])), rows)


def f1(gold, pred, strict_ask=True):
    """F1 of one prediction. `gold` and `pred` are stored execution results (`pred` None: no query).
    A query that failed (rejected, timed out, engine error) scores 0. With strict_ask, a SELECT
    prediction for an ASK question scores 0; otherwise a non-empty SELECT result counts as "yes".
    Columns are matched by GRASP's assignment, so a prediction with extra columns is not penalised."""
    if pred is None or pred["status"] not in ("ok", "empty"):
        return 0.0
    return float(f1_score(to_grasp(pred), to_grasp(gold), exact=EXACT_AFTER, strict_ask=strict_ask))


def gold_status(sparql, result):
    """Whether the gold query gives a usable answer: "ok", or the reason it does not: its execution
    status (empty, query_error, timeout, engine_error), "count_zero" for a COUNT that returns 0,
    "count_unreadable" for a COUNT query whose result is not a single number, "too_many_rows" for more
    than 1,024 rows, or "not_executed"."""
    if result is None:
        return "not_executed"
    if result["status"] != "ok":
        return result["status"]
    if TOP_LEVEL_COUNT.match(PREFIX_DECL.sub("", sparql)):
        rows = result.get("rows") or []
        try:
            (cell,) = rows[0].values() if len(rows) == 1 else ()
            count = float(cell["value"])
        except (ValueError, KeyError, TypeError):
            return "count_unreadable"
        if count == 0:
            return "count_zero"
    if (result.get("n_rows") or 0) > MAX_GOLD_ROWS:
        return "too_many_rows"
    return "ok"


def with_gold_limit(pred_query, gold_query):
    """Secondary metric for ORDER BY ... LIMIT questions. If both queries end in an ORDER BY and the
    gold query has a LIMIT, return the prediction with the gold LIMIT (replaced or appended);
    otherwise None. Only the outermost clauses count."""
    if not pred_query or not gold_query:
        return None
    gold_tail = gold_query[gold_query.rfind("}") + 1:] if "}" in gold_query else ""
    limit = re.search(r"\blimit\s+(\d+)", gold_tail, re.I)
    if not limit or not re.search(r"\border\s+by\b", gold_tail, re.I):
        return None
    cut = pred_query.rfind("}") + 1
    head, tail = pred_query[:cut], pred_query[cut:]
    if cut == 0 or not re.search(r"\border\s+by\b", tail, re.I):
        return None
    if re.search(r"\blimit\s+\d+", tail, re.I):
        tail = re.sub(r"\blimit\s+\d+", f"LIMIT {limit.group(1)}", tail, count=1, flags=re.I)
    else:
        tail = tail.rstrip() + f" LIMIT {limit.group(1)}"
    return head + tail
