import json

import pytest
import requests

from kgqa.execute import EXPORT_MARK, EndpointRefused, Executor, load_executions, query_key


class Response:
    def __init__(self, status, body, headers=None):
        self.status_code, self.headers = status, headers or {}
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        return json.loads(self.text)


def executor(responses, tmp_path, monkeypatch, **kw):
    monkeypatch.setattr("time.sleep", lambda s: None)
    ex = Executor("http://endpoint", store=str(tmp_path / "store.jsonl"), min_interval_s=0, **kw)
    queue = list(responses)

    def post(url, data=None, timeout=None):
        r = queue.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    ex.session.post = post
    return ex


def rows(n, total=None):
    body = {"head": {"vars": ["x"]}, "results": {"bindings": [{"x": {"type": "literal", "value": str(i)}} for i in range(n)]}}
    return {**body, "meta": {"result-size-total": total}} if total is not None else body


def test_results_are_stored_and_not_run_again(tmp_path, monkeypatch):
    ex = executor([Response(200, rows(2))], tmp_path, monkeypatch)
    first = ex.run("SELECT 1")
    assert first["status"] == "ok" and first["n_rows"] == 2 and first["key"] == query_key(" SELECT 1 ")
    assert ex.run("SELECT 1") == first                       # no second request: the queue is empty
    assert load_executions(str(tmp_path / "store.jsonl")) == {first["key"]: first}


def test_rows_are_capped_but_the_full_count_is_kept(tmp_path, monkeypatch):
    res = executor([Response(200, rows(4, total=900))], tmp_path, monkeypatch, max_rows=3).run("q")
    assert (res["n_rows"], len(res["rows"]), res["truncated"]) == (900, 3, True)
    assert executor([Response(200, rows(0))], tmp_path, monkeypatch).run("q2")["status"] == "empty"


@pytest.mark.parametrize("response, status", [
    (Response(400, {"exception": "parse error"}), "query_error"),
    (Response(429, {"exception": "Operation timed out"}), "timeout"),
    (Response(500, {"exception": "Tried to allocate 800 kB"}), "engine_error"),
    (Response(200, '{"head": ' + EXPORT_MARK + " The error message was: Operation timed out"), "timeout"),
    (Response(200, '{"head": ' + EXPORT_MARK + " The error message was: Tried to allocate 3 GB"), "engine_error"),
])
def test_definite_failures(response, status, tmp_path, monkeypatch):
    ex = executor([response], tmp_path, monkeypatch)
    assert ex.run("q")["status"] == status and len(ex.results) == 1


def test_transient_failures_are_retried_and_not_stored(tmp_path, monkeypatch):
    ex = executor([Response(503, "busy"), requests.exceptions.ConnectionError("down"), Response(200, rows(1))], tmp_path, monkeypatch)
    assert ex.run("q")["status"] == "ok"
    cancelled = Response(429, {"exception": "Sort operation was canceled, because time estimate exceeded remaining time"})
    ex = executor([cancelled] * 3, tmp_path, monkeypatch, max_retries=2)
    assert ex.run("other")["status"] == "http_429" and query_key("other") not in ex.results


def test_two_read_timeouts_count_as_a_timeout(tmp_path, monkeypatch):
    ex = executor([requests.exceptions.ReadTimeout(), requests.exceptions.ReadTimeout()], tmp_path, monkeypatch)
    assert ex.run("q")["status"] == "timeout"


def test_refused_requests_raise(tmp_path, monkeypatch):
    with pytest.raises(EndpointRefused):
        executor([Response(403, {"exception": "timeout too high"})], tmp_path, monkeypatch).run("q")
