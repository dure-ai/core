import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from cortex_mem.embed import HashEmbedder

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "grouping_report.py"


def _load():
    spec = importlib.util.spec_from_file_location("grouping_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


grouping_report = _load()

PAIRS = [
    {"a": "deploy uses vendor v2", "b": "deploy uses vendor v2", "same": True},
    {"a": "payments run on stripe", "b": "payments run on stripe", "same": True},
    {"a": "deploy uses vendor v2", "b": "deploy uses vendor v3", "same": False},
    {"a": "payments run on stripe", "b": "payments run on paypal", "same": False},
]


def _env_without_key():
    env = {k: v for k, v in os.environ.items() if k != "UPSTAGE_API_KEY"}
    return env


def _run(args, env=None):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env if env is not None else _env_without_key(),
        cwd=ROOT,
        check=False,
    )


def _write_pairs(tmp_path, pairs=PAIRS):
    path = tmp_path / "pairs.jsonl"
    path.write_text("\n".join(json.dumps(p) for p in pairs) + "\n", encoding="utf-8")
    return path


def test_identical_text_pairs_are_grouped_even_at_threshold_one():
    rows = grouping_report.report(
        [p for p in PAIRS if p["same"]], HashEmbedder(), [1.0]
    )
    assert rows[0].same_rate == 1.0
    assert rows[0].n_same == 2


def test_report_returns_one_row_per_threshold_with_pair_counts():
    thresholds = [0.5, 0.9]
    rows = grouping_report.report(PAIRS, HashEmbedder(), thresholds)
    assert [r.threshold for r in rows] == thresholds
    assert all(r.n_same == 2 and r.n_rival == 2 for r in rows)


def test_rates_are_zero_when_no_pairs_of_that_kind():
    rows = grouping_report.report(
        [p for p in PAIRS if p["same"]], HashEmbedder(), [0.5]
    )
    assert rows[0].n_rival == 0
    assert rows[0].rival_rate == 0.0


def test_row_is_frozen():
    row = grouping_report.Row(threshold=0.5, same_rate=1.0, rival_rate=0.0, n_same=1, n_rival=0)
    with pytest.raises(FrozenInstanceError):
        row.threshold = 0.9


def test_cli_prints_header_one_line_per_threshold_and_summary(tmp_path):
    path = _write_pairs(tmp_path)
    result = _run([str(path), "--embedder", "hash", "--thresholds", "0.5,0.9,1.0"])
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert lines[0].split("\t") == ["threshold", "same_rate", "rival_rate", "n_same", "n_rival"]
    rows = [line.split("\t") for line in lines[1:4]]
    assert [r[0] for r in rows] == ["0.5", "0.9", "1.0"]
    assert all(len(r) == 5 for r in rows)
    assert rows[-1][1] == "1.000"
    assert len(lines) == 6
    assert "0.01" in lines[-2]
    assert "chain" in lines[-1]  # rival_rate counts direct matches only (#38)


def test_cli_default_thresholds_are_seven(tmp_path):
    path = _write_pairs(tmp_path)
    result = _run([str(path), "--embedder", "hash"])
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert len(lines) == 1 + 7 + 2


def test_cli_upstage_without_key_exits_2_with_message(tmp_path):
    path = _write_pairs(tmp_path)
    result = _run(
        [str(path), "--embedder", "upstage", "--passage-model", "p", "--query-model", "q"]
    )
    assert result.returncode == 2
    assert "UPSTAGE_API_KEY" in result.stderr


def test_cli_upstage_without_models_exits_2(tmp_path):
    path = _write_pairs(tmp_path)
    env = _env_without_key()
    env["UPSTAGE_API_KEY"] = "test-key"
    result = _run([str(path), "--embedder", "upstage"], env=env)
    assert result.returncode == 2
    assert "--passage-model" in result.stderr or "--query-model" in result.stderr
    assert "test-key" not in result.stdout + result.stderr


def test_embedding_error_exits_1_without_leaking_the_key(tmp_path, monkeypatch, capsys):
    from cortex_mem.embed import EmbeddingError

    class FailingEmbedder:
        def __init__(self, api_key, passage_model, query_model):
            self._key = api_key

        def embed(self, texts, kind):
            raise EmbeddingError("embedding request failed with HTTP 401")

    monkeypatch.setattr(grouping_report, "UpstageEmbedder", FailingEmbedder)
    monkeypatch.setenv("UPSTAGE_API_KEY", "sk-secret-test")
    path = _write_pairs(tmp_path)
    rc = grouping_report.main(
        [str(path), "--embedder", "upstage", "--passage-model", "p", "--query-model", "q"]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert "error: embedding request failed with HTTP 401" in captured.err
    assert "sk-secret-test" not in captured.out + captured.err


def test_malformed_json_line_is_reported_with_its_line_number(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(PAIRS[0]) + "\n{not json\n", encoding="utf-8")
    result = _run([str(path), "--embedder", "hash"])
    assert result.returncode == 2
    assert f"{path}:2:" in result.stderr


def test_non_bool_same_is_rejected_and_not_coerced(tmp_path):
    path = tmp_path / "pairs.jsonl"
    bad = {"a": "x", "b": "y", "same": "false"}
    path.write_text(json.dumps(PAIRS[0]) + "\n" + json.dumps(bad) + "\n", encoding="utf-8")
    result = _run([str(path), "--embedder", "hash"])
    assert result.returncode == 2
    assert f"{path}:2:" in result.stderr
    assert "same" in result.stderr


@pytest.mark.parametrize("value", ["0.5,1.5", "abc", "-0.1", ""])
def test_bad_threshold_exits_2(tmp_path, value):
    path = _write_pairs(tmp_path)
    result = _run([str(path), "--embedder", "hash", "--thresholds", value])
    assert result.returncode == 2
    assert "threshold" in result.stderr


def test_empty_file_exits_2_and_asks_for_both_kinds(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text("", encoding="utf-8")
    result = _run([str(path), "--embedder", "hash"])
    assert result.returncode == 2
    assert "both" in result.stderr


def test_file_without_rival_pairs_exits_2(tmp_path):
    path = _write_pairs(tmp_path, [PAIRS[0], PAIRS[1]])
    result = _run([str(path), "--embedder", "hash"])
    assert result.returncode == 2
    assert "both" in result.stderr


def _offline_upstage(monkeypatch):
    """Record the Authorization header the embedder would send, then fail
    like an unreachable server; nothing touches the network."""
    import urllib.error

    from cortex_mem import embed

    sent = []

    def urlopen(request, timeout=None):
        sent.append(request.get_header("Authorization"))
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(embed.urllib.request, "urlopen", urlopen)
    return sent


def test_a_key_ending_in_a_carriage_return_is_stripped(tmp_path, monkeypatch, capsys):
    sent = _offline_upstage(monkeypatch)
    monkeypatch.setenv("UPSTAGE_API_KEY", "sk-secret-cr\r")
    rc = grouping_report.main(
        [str(_write_pairs(tmp_path)), "--embedder", "upstage", "--passage-model", "p", "--query-model", "q"]
    )
    out = capsys.readouterr()
    assert rc == 1  # the offline request failed, as an EmbeddingError
    assert sent == ["Bearer sk-secret-cr"]
    assert "sk-secret-cr" not in out.out + out.err


@pytest.mark.parametrize("key", ["sk-secret\r\nX: 1", "sk-비밀"])
def test_a_key_that_cannot_be_sent_exits_2_without_printing_it(tmp_path, monkeypatch, capsys, key):
    sent = _offline_upstage(monkeypatch)
    monkeypatch.setenv("UPSTAGE_API_KEY", key)
    rc = grouping_report.main(
        [str(_write_pairs(tmp_path)), "--embedder", "upstage", "--passage-model", "p", "--query-model", "q"]
    )
    out = capsys.readouterr()
    assert rc == 2
    assert "UPSTAGE_API_KEY" in out.err
    assert sent == []
    for part in ("sk-secret", "sk-비밀"):
        assert part not in out.out + out.err
