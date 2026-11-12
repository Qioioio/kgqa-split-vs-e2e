from kgqa.clean import family, structure
from kgqa.linking import link, rank_combinations
from kgqa.skeleton import fill, to_skeleton

LABELS = {"Q1045": {"label": "Somalia", "label_lang": "en"}, "P1082": {"label": "population", "label_lang": "en"},
          "P585": {"label": "point in time", "label_lang": "en"}}
GOLD = ("PREFIX wd: <http://www.wikidata.org/entity/> SELECT ?obj WHERE { wd:Q1045 p:P1082 ?s . ?s ps:P1082 ?obj . "
        "?s pq:P585 ?x filter(contains(YEAR(?x),'2009')) }")


def test_skeleton_round_trip():
    skeleton, slots = to_skeleton(GOLD, LABELS)
    assert skeleton == ("SELECT ?obj WHERE { wd:<<Somalia>> p:<<population>> ?s . ?s ps:<<population>> ?obj . "
                        "?s pq:<<point in time>> ?x filter(contains(YEAR(?x),'2009')) }")
    assert [wid for _, wid, _ in slots] == ["Q1045", "P1082", "P1082", "P585"]
    assert fill(skeleton, [wid for _, wid, _ in slots]) == GOLD.split(" ", 3)[3]


def test_no_skeleton_without_a_label():
    assert to_skeleton("SELECT ?x WHERE { wd:Q999 wdt:P1082 ?x }", LABELS) == (None, ["Q999"])


def test_rank_combinations_orders_by_sum_then_earlier_slots_first():
    assert rank_combinations([3, 2], 4) == [(0, 0), (1, 0), (0, 1), (2, 0)]
    assert rank_combinations([1, 1], 3) == [(0, 0)]
    assert rank_combinations([2, 0], 3) == []


def test_link_fills_first_candidates_and_keeps_retries():
    entities = {"Somalia": ["Q1045", "Q7"]}.get
    properties = {"population": ["P1082"], "area": ["P2046", "P9"]}.get
    out = link("SELECT ?x WHERE { wd:<<Somalia>> wdt:<<area>> ?x }", lambda n: entities(n) or [], lambda n: properties(n) or [], retries=2)
    assert out["query"].endswith("SELECT ?x WHERE { wd:Q1045 wdt:P2046 ?x }")
    assert [q.split("WHERE ")[1] for q in out["retry_queries"]] == ["{ wd:Q7 wdt:P2046 ?x }", "{ wd:Q1045 wdt:P9 ?x }"]
    assert [s["candidates"] for s in out["slots"]] == [["Q1045", "Q7"], ["P2046", "P9"]]


def test_link_gives_no_query_when_a_name_has_no_candidate():
    out = link("SELECT ?x WHERE { wd:<<Atlantis>> wdt:<<area>> ?x }", lambda n: [], lambda n: ["P2046"])
    assert out["query"] is None and out["slots"][0]["candidates"] == []
    assert link("no query here", lambda n: [], lambda n: [])["query"] is None


def test_structure_masks_everything_question_specific():
    a = structure("SELECT ?obj WHERE { wd:Q1 wdt:P2 ?obj . ?obj rdfs:label ?l . FILTER(CONTAINS(?l, 'x')) } LIMIT 25")
    b = structure("select ?a where { wd:Q9 wdt:P7 ?a . ?a rdfs:label ?b . filter(contains(?b, 'yz')) } limit 3")
    assert a == b and family(a) == "label filter + LIMIT"
    assert family(structure("ASK WHERE { wd:Q1 wdt:P2 wd:Q3 }")) == "ASK"


def test_label_names_include_aliases():
    from kgqa.skeleton import label_names
    names = label_names([{"id": "P1", "label": "Population", "aliases": ["inhabitants", " "]}, {"id": "Q9", "label": None, "aliases": []}])
    assert names == {"P1": {"population", "inhabitants"}, "Q9": set()}


def test_property_ties_are_ordered_by_training_frequency(tmp_path):
    import collections
    import gzip
    import json

    from kgqa.linking import PropertyIndex
    path = tmp_path / "labels.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in ({"id": "P5", "label": "part of", "aliases": []}, {"id": "P7", "label": "part of", "aliases": ["member of"]},
                    {"id": "P3", "label": "member of", "aliases": []}):
            f.write(json.dumps(row) + "\n")
    index = PropertyIndex(str(path), collections.Counter({"P7": 3, "P5": 1}))
    assert index.candidates("Part Of") == ["P7", "P5"]
    assert index.candidates("member of") == ["P3", "P7"]          # exact label before alias
    assert index.with_frequencies(collections.Counter({"P5": 9})).candidates("part of") == ["P5", "P7"]
    assert index.candidates("part of") == ["P7", "P5"]            # the original index is unchanged
