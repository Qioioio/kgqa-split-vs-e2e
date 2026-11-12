"""Answer-set F1 from GRASP, copied without changes.

Source: https://github.com/ad-freiburg/grasp, commit 6aa8931dee9f7c2321832bfbf6fcb65f65d4f8bd
  - AskResult, Binding, SelectResult: src/grasp/sparql/types.py
  - exact_f1_score, assignment_f1_score, f1_score: src/grasp/sparql/metrics.py
Licence: Apache License 2.0, copyright the GRASP authors (see NOTICE).
Only the imports were put together; the classes and functions are verbatim.
"""
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class AskResult:
    boolean: bool

    def __len__(self) -> int:
        return 1

    @property
    def is_empty(self) -> bool:
        return False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AskResult):
            return False

        return self.boolean == other.boolean


@dataclass
class Binding:
    typ: str
    value: str
    datatype: str | None = None
    lang: str | None = None

    def __hash__(self) -> int:
        return hash((self.typ, self.value, self.datatype, self.lang))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Binding):
            return False

        return (
            self.typ == other.typ
            and self.value == other.value
            and self.datatype == other.datatype
            and self.lang == other.lang
        )

    @staticmethod
    def from_dict(data: dict) -> "Binding":
        # virtuoso emits the pre-2013 "typed-literal" where the spec uses "literal"
        typ = data["type"]
        if typ == "typed-literal":
            typ = "literal"

        return Binding(
            typ=typ,
            value=data["value"],
            datatype=data.get("datatype"),
            lang=data.get("xml:lang"),
        )

    def identifier(self) -> str:
        assert self.typ in ["uri", "literal", "bnode"]
        match self.typ:
            case "uri":
                return self.value
            case "literal":
                if self.datatype is not None:
                    return f'"{self.value}"^^<{self.datatype}>'
                elif self.lang is not None:
                    return f'"{self.value}"@{self.lang}'
                else:
                    return f'"{self.value}"'
            case "bnode":
                return f"_:{self.value}"
            case _:
                raise ValueError(f"Unknown binding type: {self.typ}")

    def sparql(self) -> str:
        identifier = self.identifier()

        if self.typ == "uri":
            identifier = f"<{identifier}>"

        return identifier

    def __repr__(self) -> str:
        return self.identifier()


SelectRow = dict[str, Binding]


@dataclass
class SelectResult:
    variables: list[str]
    data: list[dict | None]
    complete: bool = True

    @staticmethod
    def from_json(data: dict) -> "SelectResult":
        return SelectResult(
            variables=data["head"]["vars"],
            data=data["results"]["bindings"],
        )

    def truncate(self, max_rows: int) -> None:
        if len(self) <= max_rows:
            return

        self.data = self.data[:max_rows]
        self.complete = False

    def __len__(self) -> int:
        return len(self.data)

    def bindings(
        self,
        start: int = 0,
        end: int | None = None,
    ) -> Iterator[tuple[Binding, ...]]:
        for row in self.rows(start, end):
            bindings = tuple(row[var] for var in self.variables if var in row)
            yield bindings

    def rows(self, start: int = 0, end: int | None = None) -> Iterator[SelectRow]:
        start = max(start, 0)

        if end is None:
            end = len(self.data)
        else:
            end = min(end, len(self.data))

        for i in range(start, end):
            data = self.data[i]
            if data is None:
                yield {}
            else:
                yield {
                    var: Binding.from_dict(data[var])
                    for var in self.variables
                    if var in data
                }

    @property
    def num_rows(self) -> int:
        return len(self.data)

    @property
    def num_columns(self) -> int:
        return len(self.variables)

    @property
    def is_empty(self) -> bool:
        return not self.data

    def to_ask_result(self) -> AskResult:
        return AskResult(not self.is_empty)


def exact_f1_score(pred: Iterable[tuple], target: Iterable[tuple]) -> float:
    pred_set = Counter(pred)
    target_set = Counter(target)

    tp = (pred_set & target_set).total()
    if tp == 0:
        return 0.0

    fp = (pred_set - target_set).total()
    fn = (target_set - pred_set).total()
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def assignment_f1_score(
    pred: Iterable[Iterable],
    target: Iterable[Iterable],
) -> float:
    # create a matrix of distances between pred and target
    pred = [Counter(p) for p in pred]  # type: ignore
    target = [Counter(t) for t in target]  # type: ignore

    scores = np.zeros((len(pred), len(target)), dtype=np.float32)

    for i, p_set in enumerate(pred):
        for j, t_set in enumerate(target):
            r = (p_set & t_set).total() / max(1, t_set.total())
            scores[i, j] = r

    rows, cols = linear_sum_assignment(scores, maximize=True)
    assert len(rows) == len(cols) == min(len(pred), len(target))
    assignment_scores = scores[rows, cols]
    tp = float(assignment_scores.sum())
    fn = float((1 - assignment_scores).sum()) + len(target) - len(rows)
    fp = len(pred) - len(rows)
    if tp <= 0.0:
        return 0.0

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def f1_score(
    pred: SelectResult | AskResult,
    target: SelectResult | AskResult,
    exact: int | bool = 1024,
    strict_ask: bool = False,
) -> float:
    # if reference is an ask query, convert the select results to a bool; with
    # strict_ask a select prediction scores 0 instead, since a non-empty result
    # also counts as "yes" when the query ignores the question's condition
    if isinstance(target, AskResult):
        if strict_ask and not isinstance(pred, AskResult):
            return 0.0
        pred = pred.to_ask_result() if not isinstance(pred, AskResult) else pred
        return float(pred == target)

    # if reference is a select query, a ask prediction cannot be
    # judged correctly, so return 0.0; if we turn the reference into a bool
    # a trivial ASK query like ASK { ?s ?p ?o } would always score perfectly
    if isinstance(pred, AskResult):
        return 0.0

    if pred.is_empty and target.is_empty:
        return 1.0
    elif pred.is_empty or target.is_empty:
        return 0.0

    if isinstance(exact, bool):
        exact_after = 0 if exact else max(len(pred), len(target))
    else:
        exact_after = exact

    if len(pred) > exact_after or len(target) > exact_after:
        return exact_f1_score(pred.bindings(), target.bindings())
    else:
        return assignment_f1_score(pred.bindings(), target.bindings())
