from kgqa.execute import query_key
from kgqa.scoring import score_prediction

E = "http://www.wikidata.org/entity/"
YES = {"status": "ok", "n_rows": 1, "boolean": True}


def select(*values):
    rows = [{"x": {"type": "uri", "value": E + v}} for v in values]
    return {"status": "ok" if rows else "empty", "n_rows": len(rows), "vars": ["x"], "rows": rows}


def lookup(table):
    return lambda query: table.get(query_key(query))


def test_strict_and_lenient_ask():
    pred = {"query": "SELECT ?x WHERE { ?x ?p ?o }"}
    s = score_prediction(pred, "ASK { ?s ?p ?o }", YES, lookup({query_key(pred["query"]): select("Q1")}))
    assert (s["f1"], s["f1_lenient_ask"], s["f1_gold_limit"]) == (0.0, 1.0, 0.0)


def test_retry_queries_are_tried_only_while_the_result_is_empty():
    gold = select("Q2")
    pred = {"query": "q0", "retry_queries": ["q1", "q2", "q3"]}
    table = {query_key("q0"): select(), query_key("q1"): select(), query_key("q2"): select("Q2"), query_key("q3"): select("Q9")}
    s = score_prediction(pred, "SELECT ?x WHERE { }", gold, lookup(table))
    assert (s["f1"], s["f1_retry"]) == (0.0, 1.0)
    table[query_key("q0")] = select("Q7")                      # a non-empty first result is final
    assert score_prediction(pred, "SELECT ?x WHERE { }", gold, lookup(table))["f1_retry"] == 0.0


def test_gold_limit_variant_and_missing_results():
    gold_query = "SELECT ?x WHERE { ?x ?p ?o } ORDER BY ?o LIMIT 1"
    pred = {"query": "SELECT ?x WHERE { ?x ?p ?o } ORDER BY ?o"}
    limited = "SELECT ?x WHERE { ?x ?p ?o } ORDER BY ?o LIMIT 1"
    table = {query_key(pred["query"]): select("Q1", "Q2"), query_key(limited): select("Q1")}
    s = score_prediction(pred, gold_query, select("Q1"), lookup(table))
    assert s["f1"] < 1.0 and s["f1_gold_limit"] == 1.0
    del table[query_key(limited)]
    assert score_prediction(pred, gold_query, select("Q1"), lookup(table)) is None     # a needed result is missing
    assert score_prediction({"query": None}, gold_query, select("Q1"), lookup(table))["f1"] == 0.0
