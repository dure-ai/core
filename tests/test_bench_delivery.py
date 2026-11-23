import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pg import pg_url

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bench_delivery.py"


def _load():
    spec = importlib.util.spec_from_file_location("bench_delivery", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load()


def _check_shape(result, store, events, hold):
    assert set(result) == {
        "label",
        "store",
        "commit",
        "events",
        "write_transactions",
        "embedder_calls",
        "append_while_observe_holds",
    }
    assert result["label"] == "t"
    assert result["store"] == store
    assert isinstance(result["commit"], str) and result["commit"]
    assert result["events"] == events
    wt = result["write_transactions"]
    assert set(wt) == {"total", "per_event"}
    assert isinstance(wt["total"], int) and wt["total"] >= events
    assert isinstance(wt["per_event"], float)
    assert isinstance(result["embedder_calls"], int) and result["embedder_calls"] > 0
    held = result["append_while_observe_holds"]
    assert set(held) == {"hold_s", "raw_append_s", "runtime_append_s", "errors"}
    assert held["hold_s"] == hold
    assert isinstance(held["errors"], int)
    for key in ("raw_append_s", "runtime_append_s"):
        assert set(held[key]) == {"min", "median", "max"}
        if held["errors"] == 0:
            assert 0 <= held[key]["min"] <= held[key]["median"] <= held[key]["max"]
    json.dumps(result)  # serializable


def test_sqlite_result_shape():
    result = bench.run_bench("sqlite", "t", events=5, hold=0.2, reps=2)
    _check_shape(result, "sqlite", 5, 0.2)


def test_pg_result_shape(pg_schema):
    if pg_url() is None:
        pytest.skip("CORTEX_TEST_PG_URL is not set")
    result = bench.run_bench("pg", "t", events=5, hold=0.2, reps=2, pg_url=pg_schema)
    _check_shape(result, "pg", 5, 0.2)


def test_main_writes_json_file(tmp_path, capsys):
    out = tmp_path / "nested" / "b.json"
    code = bench.main(["--store", "sqlite", "--label", "t", "--events", "3", "--hold", "0.1", "--out", str(out)])
    assert code == 0
    written = json.loads(out.read_text())
    assert written["events"] == 3
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1]) == written


def test_main_without_out_prints_and_writes_no_file(monkeypatch, capsys):
    written = []
    monkeypatch.setattr(Path, "write_text", lambda self, *a, **kw: written.append(self))
    code = bench.main(["--store", "sqlite", "--label", "t", "--events", "3", "--hold", "0.1"])
    assert code == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["events"] == 3
    assert written == []  # a default run leaves nothing in the checkout


def test_bad_store_exits_2_with_message():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--store", "mysql", "--label", "t"], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 2
    assert "--store" in proc.stderr


def test_pg_without_url_exits_2(monkeypatch):
    monkeypatch.delenv("CORTEX_BENCH_PG_URL", raising=False)
    with pytest.raises(SystemExit) as exc:
        bench.main(["--store", "pg", "--label", "t"])
    assert exc.value.code == 2
