"""Run SPARQL queries on one endpoint and keep the results.

One request at a time, at least a second apart, with a 30 s server-side timeout.
Every query ends in one of five definite outcomes, which are stored and never re-run:

    ok           a non-empty result (at most `max_rows` rows are kept; `n_rows` is the full count)
    empty        no rows
    query_error  the endpoint rejected the query (HTTP 400)
    timeout      the endpoint gave up on the query, or the connection stayed silent past the
                 timeout twice
    engine_error the endpoint failed on the query (HTTP 500 with an error message, or an error
                 note appended to a response that had already started)

Anything else (connection problems, 429/5xx without a verdict on the query) is retried with
backoff and, if it persists, returned but not stored, so that a later run tries again.

The server options `timeout` and `send` and the handling of errors in the middle of a
response are specific to QLever.
"""
import gzip
import hashlib
import json
import os
import re
import time

import requests

USER_AGENT = "kgqa-split-vs-e2e/1.0 (https://github.com/Qioioio/kgqa-split-vs-e2e; research evaluation, one request at a time)"
QLEVER_WIKIDATA = "https://qlever.dev/api/wikidata"
DEFINITE = {"ok", "empty", "query_error", "timeout", "engine_error"}
RETRY_HTTP = {429, 500, 502, 503, 504}
TIMEOUT_MSG = re.compile(r"time[ds]?\s*out|timeout", re.I)
CANCEL_MSG = re.compile(r"cancel", re.I)
# QLever streams results: when it fails after the first rows have gone out, it has already sent
# HTTP 200 and can only append this note to the unfinished JSON.
EXPORT_MARK = "!!!!>># An error has occurred while exporting the query result."
EXPORT_NOTE = re.compile(re.escape(EXPORT_MARK) + r".*?The error message was:(.*)\Z", re.S)
PING = "SELECT ?x WHERE { VALUES ?x { 1 } }"


class EndpointRefused(RuntimeError):
    """HTTP 401/403: the endpoint rejects our settings or credentials. Not a result of the query."""


def query_key(query):
    return hashlib.sha256(query.strip().encode("utf-8")).hexdigest()


def load_executions(path):
    """{key: record} from a JSON-lines file (plain or .gz)."""
    opener = gzip.open if path.endswith(".gz") else open
    out = {}
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[rec["key"]] = rec
    return out


def save_executions(path, records):
    """Write the records sorted by key, gzipped, byte-identical for identical content."""
    with gzip.GzipFile(path, "wb", mtime=0) as f:
        for key in sorted(records):
            f.write((json.dumps(records[key], ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))


def export_error_message(body):
    """The error message QLever appended to a broken response; None if there is no such note."""
    i = body.rfind(EXPORT_MARK)
    if i < 0:
        return None
    m = EXPORT_NOTE.match(body, i)
    return None if m is None else m.group(1).strip()


class Executor:
    def __init__(self, url=QLEVER_WIKIDATA, store=None, qlever=True, timeout_s=30, max_rows=5000,
                 min_interval_s=1.0, max_retries=6):
        self.url, self.qlever = url, qlever
        self.timeout_s, self.max_rows = timeout_s, max_rows
        self.min_interval_s, self.max_retries = min_interval_s, max_retries
        self.store = store   # JSON-lines file: definite results are appended as they arrive, so a run can be resumed
        self.results = load_executions(store) if store and os.path.exists(store) else {}
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"})
        self.last = 0.0

    def _wait(self):
        gap = time.monotonic() - self.last
        if gap < self.min_interval_s:
            time.sleep(self.min_interval_s - gap)

    def _form(self, query):
        form = {"query": query}
        if self.qlever:
            form["timeout"] = f"{int(self.timeout_s * 1000)}ms"   # the server cancels the query itself
            form["send"] = str(self.max_rows + 1)                 # and sends back only the first rows
        return form

    def _keep(self, key, record):
        record = {"key": key, **record}
        self.results[key] = record
        if self.store:
            with open(self.store, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        return record

    def _verdict(self, r):
        """The definite outcome this response gives for the query, or None if it gives none."""
        code = r.status_code
        try:
            js = r.json()
        except ValueError:
            js = None
        if code == 200 and isinstance(js, dict) and ("boolean" in js or isinstance(js.get("results"), dict)):
            if "boolean" in js:
                return {"status": "ok", "n_rows": 1, "boolean": bool(js["boolean"])}
            rows = (js.get("results") or {}).get("bindings") or []
            n_rows = len(rows)
            out = {"status": "ok" if n_rows else "empty", "n_rows": n_rows,
                   "vars": (js.get("head") or {}).get("vars", []), "rows": rows[: self.max_rows]}
            if n_rows > self.max_rows:
                out["truncated"] = True
            total = js["meta"].get("result-size-total") if isinstance(js.get("meta"), dict) else None
            if isinstance(total, int) and total > n_rows:   # with `send`, the full count comes from the server
                out["n_rows"] = total
            return out
        if code in (401, 403):
            raise EndpointRefused(f"HTTP {code}: {r.text[:300]}")
        if code == 200 and js is None:
            msg = export_error_message(r.text)
            if not msg or (CANCEL_MSG.search(msg) and not TIMEOUT_MSG.search(msg)):
                return None
            msg = "QLever export error (HTTP 200): " + msg
            return {"status": "timeout" if TIMEOUT_MSG.search(msg) else "engine_error", "message": msg[:300]}
        exception = js.get("exception") if isinstance(js, dict) else None
        if code == 400:
            return {"status": "query_error", "message": (str(exception) if exception is not None else r.text)[:300]}
        if exception is not None and code in (429, 500):
            if TIMEOUT_MSG.search(str(exception)):
                return {"status": "timeout", "message": str(exception)[:300]}
            if code == 500:
                return {"status": "engine_error", "message": str(exception)[:300]}
            # A 429 that does not mention a timeout means the query was cancelled, e.g. by a restart.
            # (QLever also answers this way when it estimates that a sort cannot finish in time.)
        return None

    def run(self, query):
        """The stored record for `query`, executing it first if needed. A record whose status is not
        in DEFINITE was not stored."""
        key = query_key(query)
        if key in self.results:
            return self.results[key]
        form, attempts, read_timeouts, backoff = self._form(query), 0, 0, 2.0
        while True:
            attempts += 1
            self._wait()
            try:
                r = self.session.post(self.url, data=form, timeout=(15, self.timeout_s + 10))
            except requests.exceptions.ReadTimeout:
                # the server cancels at timeout_s on its own; silence for this long is more likely a
                # stuck connection, so try once more before calling it a timeout
                self.last = time.monotonic()
                read_timeouts += 1
                if read_timeouts < 2:
                    continue
                return self._keep(key, {"status": "timeout", "message": f"client read timeout ({self.timeout_s + 10:g} s) twice"})
            except requests.exceptions.RequestException as e:
                self.last = time.monotonic()
                if attempts <= self.max_retries:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 120)
                    continue
                return {"key": key, "status": "connection_error", "message": f"{type(e).__name__}: {str(e)[:200]}"}
            self.last = time.monotonic()
            verdict = self._verdict(r)
            if verdict is not None:
                return self._keep(key, verdict)
            if (r.status_code in RETRY_HTTP or r.status_code == 200) and attempts <= self.max_retries:
                retry_after = r.headers.get("Retry-After", "")
                time.sleep(min(float(retry_after), 300) if retry_after.isdigit() else backoff)
                backoff = min(backoff * 2, 120)
                continue
            return {"key": key, "status": f"http_{r.status_code}", "message": r.text[:300]}

    def ping(self):
        """True if the endpoint answers a trivial query with the settings used for real ones."""
        self._wait()
        try:
            r = self.session.post(self.url, data=self._form(PING), timeout=(15, 30))
        except requests.exceptions.RequestException:
            return False
        finally:
            self.last = time.monotonic()
        if r.status_code in (401, 403):
            raise EndpointRefused(f"HTTP {r.status_code}: {r.text[:300]}")
        return r.status_code == 200 and '"results"' in r.text

    def index_info(self):
        """What QLever reports about the loaded index (?cmd=stats and ?cmd=get-index-id), to check that
        it did not change during a run. None for other endpoints."""
        if not self.qlever:
            return None
        out = {}
        for cmd in ("stats", "get-index-id"):
            self._wait()
            try:
                r = self.session.get(self.url, params={"cmd": cmd}, timeout=(15, 30))
                try:
                    out[cmd] = r.json()
                except ValueError:
                    out[cmd] = r.text[:2000]
            except requests.exceptions.RequestException as e:
                out[cmd] = f"{type(e).__name__}: {str(e)[:200]}"
            self.last = time.monotonic()
        return out
