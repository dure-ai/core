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

from cortex_mem.embed import Embedder, HashEmbedder, UpstageEmbedder, cosine

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
    return [float(part) for part in text.split(",") if part.strip()]


def _read_pairs(path: Path) -> list[dict[str, Any]]:
    pairs = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                pairs.append(json.loads(line))
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

    if args.embedder == "hash":
        embedder: Embedder = HashEmbedder()
    else:
        api_key = os.environ.get(KEY_ENV, "")
        if not api_key:
            print(f"error: set {KEY_ENV} to use the upstage embedder", file=sys.stderr)
            return 2
        if not args.passage_model or not args.query_model:
            print("error: --passage-model and --query-model are required for upstage", file=sys.stderr)
            return 2
        embedder = UpstageEmbedder(api_key, args.passage_model, args.query_model)

    rows = report(_read_pairs(args.pairs), embedder, _parse_thresholds(args.thresholds))
    print(_format(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
