"""Report how labelled text pairs would group at each cosine threshold.

Input is JSON Lines, one object per line: {"a": str, "b": str, "same": bool}.
A pair is grouped at threshold t when cosine(a, b) >= t. For each threshold the
report prints the share of "same" pairs and "rival" pairs that are grouped.

Usage:
    python scripts/grouping_report.py PAIRS.jsonl --embedder hash
    UPSTAGE_API_KEY=... python scripts/grouping_report.py PAIRS.jsonl \
        --embedder upstage --passage-model M --query-model M
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cortex_mem.embed import (
    Embedder,
    EmbeddingError,
    HashEmbedder,
    UpstageEmbedder,
    cosine,
)

DEFAULT_THRESHOLDS = "0.5,0.6,0.7,0.8,0.85,0.9,0.95"
RIVAL_LIMIT = 0.01
KEY_ENV = "UPSTAGE_API_KEY"


@dataclass(frozen=True)
class Row:
    """Grouping rates for one threshold."""

    threshold: float
    same_rate: float
    rival_rate: float
    n_same: int
    n_rival: int


def report(pairs: Sequence[dict[str, Any]], embedder: Embedder, thresholds: Sequence[float]) -> list[Row]:
    """Embed every distinct text once, then compute the grouping rates per threshold."""
    texts = list(dict.fromkeys(text for p in pairs for text in (p["a"], p["b"])))
    vectors = dict(zip(texts, embedder.embed(texts, "passage")))
    sims = [(cosine(vectors[p["a"]], vectors[p["b"]]), bool(p["same"])) for p in pairs]
    n_same = sum(1 for _, same in sims if same)
    n_rival = len(sims) - n_same
    rows = []
    for t in thresholds:
        same_hit = sum(1 for sim, same in sims if same and sim >= t)
        rival_hit = sum(1 for sim, same in sims if not same and sim >= t)
        rows.append(
            Row(
                threshold=t,
                same_rate=same_hit / n_same if n_same else 0.0,
                rival_rate=rival_hit / n_rival if n_rival else 0.0,
                n_same=n_same,
                n_rival=n_rival,
            )
        )
    return rows


def _parse_thresholds(text: str) -> list[float]:
    thresholds = []
    for part in text.split(","):
        if not part.strip():
            continue
        try:
            value = float(part)
        except ValueError:
            raise ValueError(f"threshold {part!r} is not a number") from None
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"threshold {value} is outside 0 to 1")
        thresholds.append(value)
    if not thresholds:
        raise ValueError("--thresholds needs at least one value")
    return thresholds


def _read_pairs(path: Path) -> list[dict[str, Any]]:
    pairs = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            where = f"{path}:{lineno}"
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"{where}: not valid JSON") from None
            if not isinstance(obj, dict):
                raise ValueError(f"{where}: expected a JSON object")  # noqa: TRY004
            if not isinstance(obj.get("a"), str) or not isinstance(obj.get("b"), str):
                raise ValueError(f"{where}: 'a' and 'b' must be strings")  # noqa: TRY004
            if not isinstance(obj.get("same"), bool):
                raise ValueError(f"{where}: 'same' must be true or false")  # noqa: TRY004
            pairs.append(obj)
    return pairs


def _format(rows: list[Row]) -> str:
    lines = ["threshold\tsame_rate\trival_rate\tn_same\tn_rival"]
    for r in rows:
        lines.append(f"{r.threshold}\t{r.same_rate:.3f}\t{r.rival_rate:.3f}\t{r.n_same}\t{r.n_rival}")
    ok = [r.threshold for r in rows if r.rival_rate <= RIVAL_LIMIT]
    if ok:
        lines.append(f"lowest threshold with rival_rate <= {RIVAL_LIMIT}: {min(ok)}")
    else:
        lines.append(f"no threshold qualifies with rival_rate <= {RIVAL_LIMIT}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pairs", type=Path, help="JSON Lines file of {a, b, same} objects")
    parser.add_argument("--embedder", choices=["hash", "upstage"], required=True)
    parser.add_argument("--passage-model")
    parser.add_argument("--query-model")
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    args = parser.parse_args(argv)

    try:
        thresholds = _parse_thresholds(args.thresholds)
        pairs = _read_pairs(args.pairs)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    n_same = sum(1 for p in pairs if p["same"])
    n_rival = len(pairs) - n_same
    if n_same == 0 or n_rival == 0:
        print(
            f"error: the measurement needs both same and rival pairs "
            f"(found {n_same} same, {n_rival} rival)",
            file=sys.stderr,
        )
        return 2

    if args.embedder == "hash":
        embedder: Embedder = HashEmbedder()
    else:
        api_key = os.environ.get(KEY_ENV, "").strip()
        if not api_key:
            print(f"error: set {KEY_ENV} to use the upstage embedder", file=sys.stderr)
            return 2
        if not args.passage_model or not args.query_model:
            print("error: --passage-model and --query-model are required for upstage", file=sys.stderr)
            return 2
        try:
            embedder = UpstageEmbedder(api_key, args.passage_model, args.query_model)
        except ValueError:
            # The constructor's message never quotes the key; this one does not either.
            print(f"error: {KEY_ENV} is not a valid key (printable ASCII only)", file=sys.stderr)
            return 2

    try:
        rows = report(pairs, embedder, thresholds)
    except EmbeddingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(_format(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
