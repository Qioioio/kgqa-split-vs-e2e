from kgqa.postprocess import extract_query


def test_plain_query_gets_the_prefixes_it_uses():
    query, flags = extract_query("SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }")
    assert query.splitlines() == ["PREFIX wd: <http://www.wikidata.org/entity/>",
                                  "PREFIX wdt: <http://www.wikidata.org/prop/direct/>", "SELECT ?x WHERE { wd:Q1 wdt:P31 ?x }"]
    assert flags["prefixes_added"] == ["wd", "wdt"]


def test_query_is_taken_out_of_a_code_fence_and_prose():
    text = "Here is the query:\n```sparql\nSELECT ?x WHERE { ?x ?p ?o } LIMIT 3\n```\nIt selects everything."
    assert extract_query(text)[0] == "SELECT ?x WHERE { ?x ?p ?o } LIMIT 3"
    assert extract_query("Sure. Select the right one:\nASK { ?s ?p ?o }")[0] == "ASK { ?s ?p ?o }"


def test_text_after_the_query_is_dropped_but_solution_modifiers_are_kept():
    query, flags = extract_query("SELECT ?x WHERE { ?x ?p ?o }\nORDER BY ?x\nLIMIT 2\nSELECT ?y WHERE { ?y ?p ?o }")
    assert query == "SELECT ?x WHERE { ?x ?p ?o }\nORDER BY ?x\nLIMIT 2" and flags["multi_query"]


def test_nothing_to_extract():
    assert extract_query("")[0] is None
    assert extract_query("I cannot answer that.")[0] is None


def test_unbalanced_braces_are_flagged():
    query, flags = extract_query("SELECT ?x WHERE { ?x ?p ?o ")
    assert flags["unbalanced"] and query.startswith("SELECT")


def test_label_service_is_replaced_by_optional_labels():
    query, flags = extract_query('SELECT ?x ?xLabel WHERE { ?x wdt:P31 wd:Q5 . SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } }')
    assert flags["label_service_rewritten"] == 1 and "SERVICE" not in query
    assert 'OPTIONAL { ?x rdfs:label ?xLabel . FILTER (LANG(?xLabel) = "en") }' in query
    assert "PREFIX rdfs:" in query


def test_declared_prefixes_are_kept_and_conflicts_reported():
    query, flags = extract_query("PREFIX wd: <http://example.org/>\nSELECT ?x WHERE { wd:Q1 ?p ?x }")
    assert query.startswith("PREFIX wd: <http://example.org/>") and flags["prefix_conflicts"] == ["wd"]
