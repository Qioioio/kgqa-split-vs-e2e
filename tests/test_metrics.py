from kgqa.metrics import f1, gold_status, normalize_term, with_gold_limit
from kgqa.stats import holm, mean_ci, paired_test

E = "http://www.wikidata.org/entity/"
XSD = "http://www.w3.org/2001/XMLSchema#"


def select(*values, var="x"):
    rows = [{var: {"type": "uri", "value": E + v}} for v in values]
    return {"status": "ok" if rows else "empty", "n_rows": len(rows), "vars": [var], "rows": rows}


def test_f1_compares_answer_sets_whatever_the_variable_is_called():
    assert f1(select("Q1", "Q2"), select("Q1", "Q2", var="answer")) == 1.0
    assert f1(select("Q1", "Q2"), select("Q1")) == 2 * 1.0 * 0.5 / 1.5
    assert f1(select("Q1"), select("Q3")) == 0.0
    assert f1(select("Q1"), select()) == 0.0


def test_no_query_and_failed_queries_score_zero():
    assert f1(select("Q1"), None) == 0.0
    assert f1(select("Q1"), {"status": "timeout"}) == 0.0
    assert f1(select("Q1"), {"status": "query_error", "message": "..."}) == 0.0


def test_ask_questions_strict_and_lenient():
    yes = {"status": "ok", "n_rows": 1, "boolean": True}
    assert f1(yes, {"status": "ok", "n_rows": 1, "boolean": True}) == 1.0
    assert f1(yes, {"status": "ok", "n_rows": 1, "boolean": False}) == 0.0
    assert f1(yes, select("Q1")) == 0.0                       # a SELECT for an ASK question
    assert f1(yes, select("Q1"), strict_ask=False) == 1.0     # ... counts as "yes" under the lenient reading


def test_literals_are_compared_by_value():
    def lit(value, datatype=None, lang=None):
        return normalize_term({"type": "literal", "value": value, **({"datatype": datatype} if datatype else {}),
                               **({"xml:lang": lang} if lang else {})})
    assert lit("3.0", XSD + "decimal") == lit("3", XSD + "integer")
    assert lit("2001-05-04T00:00:00Z", XSD + "dateTime") == lit("2001-05-04", XSD + "date")
    assert lit("Paris", lang="en") == lit("Paris", XSD + "string") == lit("Paris")
    assert lit("2001-05-04T10:00:00Z", XSD + "dateTime") != lit("2001-05-04", XSD + "date")


def test_gold_status():
    count = "SELECT (COUNT(?x) AS ?n) WHERE { ?x wdt:P31 wd:Q5 }"
    zero = {"status": "ok", "n_rows": 1, "vars": ["n"], "rows": [{"n": {"type": "literal", "value": "0"}}]}
    assert gold_status("SELECT ?x WHERE { }", select("Q1")) == "ok"
    assert gold_status("SELECT ?x WHERE { }", select()) == "empty"
    assert gold_status(count, zero) == "count_zero"
    assert gold_status("SELECT ?x WHERE { }", {**select("Q1"), "n_rows": 1025}) == "too_many_rows"
    assert gold_status("SELECT ?x WHERE { }", None) == "not_executed"


def test_with_gold_limit_only_touches_order_by_queries():
    gold = "SELECT ?x WHERE { ?x wdt:P1 ?y } ORDER BY DESC(?y) LIMIT 5"
    assert with_gold_limit("SELECT ?x WHERE { ?x wdt:P1 ?y } ORDER BY DESC(?y)", gold).endswith("ORDER BY DESC(?y) LIMIT 5")
    assert with_gold_limit("SELECT ?x WHERE { ?x wdt:P1 ?y } ORDER BY ?y LIMIT 1", gold).endswith("ORDER BY ?y LIMIT 5")
    assert with_gold_limit("SELECT ?x WHERE { ?x wdt:P1 ?y } LIMIT 1", gold) is None
    assert with_gold_limit("SELECT ?x WHERE { ?x wdt:P1 ?y } ORDER BY ?y", "SELECT ?x WHERE { ?x wdt:P1 ?y } LIMIT 5") is None


def test_statistics():
    a, b = [1.0] * 60 + [0.0] * 40, [1.0] * 30 + [0.0] * 70
    mean, (low, high) = mean_ci(a)
    assert mean == 60.0 and low < 60 < high
    t = paired_test(a, b)
    assert t["diff"] == 30.0 and t["ci95"][0] > 0 and t["p"] < 0.001
    assert paired_test(a, a)["p"] == 1.0
    adjusted = holm({"x": 0.01, "y": 0.04, "z": 0.03})
    assert [round(adjusted[k][0], 2) for k in "xyz"] == [0.03, 0.06, 0.06] and [adjusted[k][1] for k in "xyz"] == [True, False, False]
