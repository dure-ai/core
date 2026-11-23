"""Validated configuration for the ledger engine."""

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import timedelta
from types import MappingProxyType
from typing import TypeVar, cast

_T = TypeVar("_T")

_SCOPES = ("personal", "team")
_CHANNELS = ("observation", "review", "test", "doc", "human")
_STATUSES = ("candidate", "provisional", "established", "disputed")
_GATE_FIELDS = (
    "min_sources",
    "alt_verified",
    "require_verified",
    "min_span_days",
    "max_contra_ratio",
)


@dataclass(frozen=True)
class GateRule:
    """Promotion gate thresholds for one scope."""

    min_sources: int
    alt_verified: int
    require_verified: bool
    min_span: timedelta
    max_contra_ratio: float


@dataclass(frozen=True)
class LedgerConfig:
    """All tunables of the ledger engine, with validated defaults."""

    similar_same_threshold: float
    same_threshold: float
    copy_threshold: float
    match_pool: int
    pool: int
    gates: Mapping[str, GateRule]
    weights: Mapping[str, float]
    trust_observation: float
    alpha: float
    beta: float
    status_weights: Mapping[str, float]
    own_candidate_weight: float
    review_after: timedelta
    half_life: timedelta
    high_risk_terms: tuple[str, ...]

    @classmethod
    def from_mapping(cls, config: Mapping[str, str] | None) -> "LedgerConfig":
        """Build a config from string settings, refusing unknown keys and bad values."""
        raw = dict(config or {})
        _check_keys(raw)

        def get(key: str, parse: Callable[[str, str], _T], default: _T) -> _T:
            return parse(key, raw[key]) if key in raw else default

        similar = get("similar.same_threshold", _float, 0.0)
        if not 0.0 <= similar <= 1.0:
            raise ValueError("similar.same_threshold must be within 0..1")
        same = get("same_threshold", _float, 0.8)
        if not 0.0 < same <= 1.0:
            raise ValueError("same_threshold must be > 0 and <= 1")
        copy = get("copy_threshold", _float, 0.9)
        if not same <= copy <= 1.0:
            raise ValueError("copy_threshold must be >= same_threshold and <= 1")
        match_pool = get("match_pool", _int, 10)
        if match_pool < 1:
            raise ValueError("match_pool must be >= 1")
        pool = get("pool", _int, 50)
        if pool < 1:
            raise ValueError("pool must be >= 1")

        # Team knowledge needs a verified source by default: records in one claim are only known to
        # be on one topic, not to agree (there is no judge yet), so a record with another value would
        # otherwise supply the second vote.
        gate_defaults = {
            "personal": (2, 1, False, 0.0, 0.20),
            "team": (2, 1, True, 0.0, 0.15),
        }
        gates = {}
        for scope in _SCOPES:
            d = gate_defaults[scope]
            p = f"gate.{scope}."
            min_sources = get(p + "min_sources", _int, d[0])
            alt_verified = get(p + "alt_verified", _int, d[1])
            require_verified = get(p + "require_verified", _bool, d[2])
            span_days = get(p + "min_span_days", _float, d[3])
            contra = get(p + "max_contra_ratio", _float, d[4])
            if min_sources < 0:
                raise ValueError(f"{p}min_sources must be >= 0")
            if alt_verified < 0:
                raise ValueError(f"{p}alt_verified must be >= 0")
            if span_days < 0:
                raise ValueError(f"{p}min_span_days must be >= 0")
            if not 0.0 <= contra <= 1.0:
                raise ValueError(f"{p}max_contra_ratio must be within 0..1")
            gates[scope] = GateRule(
                min_sources,
                alt_verified,
                require_verified,
                timedelta(days=span_days),
                contra,
            )

        weight_defaults = {
            "observation": 1.0,
            "review": 1.5,
            "test": 2.0,
            "doc": 2.0,
            "human": 3.0,
        }
        weights = {
            ch: _non_negative(f"weight.{ch}", get(f"weight.{ch}", _float, weight_defaults[ch]))
            for ch in _CHANNELS
        }
        status_defaults = {
            "candidate": 0.5,
            "provisional": 0.6,
            "established": 1.0,
            "disputed": 0.4,
        }
        status_weights = {
            st: _non_negative(
                f"status_weight.{st}",
                get(f"status_weight.{st}", _float, status_defaults[st]),
            )
            for st in _STATUSES
        }

        alpha = _non_negative("alpha", get("alpha", _float, 1.0))
        beta = _non_negative("beta", get("beta", _float, 1.0))
        if alpha + beta == 0:
            # confidence (S + alpha) / (S + C + alpha + beta) is 0/0 with no strength
            raise ValueError("alpha and beta must not both be 0")

        terms: tuple[str, ...] = ()
        if "high_risk_terms" in raw:
            parts = (t.strip().lower() for t in raw["high_risk_terms"].split(","))
            terms = tuple(t for t in parts if t)

        return cls(
            similar_same_threshold=similar,
            same_threshold=same,
            copy_threshold=copy,
            match_pool=match_pool,
            pool=pool,
            gates=MappingProxyType(gates),
            weights=MappingProxyType(weights),
            trust_observation=_non_negative(
                "trust.observation", get("trust.observation", _float, 0.7)
            ),
            alpha=alpha,
            beta=beta,
            status_weights=MappingProxyType(status_weights),
            own_candidate_weight=_non_negative(
                "own_candidate_weight", get("own_candidate_weight", _float, 0.7)
            ),
            review_after=_days("review_after_days", get("review_after_days", _float, 0.0)),
            half_life=_days("half_life_days", get("half_life_days", _float, 0.0)),
            high_risk_terms=terms,
        )

    def fingerprint(self) -> str:
        """A short digest of every parsed setting. Equal settings give equal
        digests however they were written ("0.8" and "0.80")."""
        text = json.dumps(_plain(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()[:12]


def _plain(value: object) -> object:
    """`value` as JSON-ready data: dataclasses and mappings as dicts,
    timedeltas as seconds, tuples as lists."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(cast(object, getattr(value, f.name))) for f in fields(value)}
    if isinstance(value, Mapping):
        items = cast("Mapping[object, object]", value).items()
        return {str(k): _plain(v) for k, v in items}
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, tuple):
        return [_plain(v) for v in cast("tuple[object, ...]", value)]
    return value


def _known_keys() -> set[str]:
    """Every config key the engine accepts."""
    keys = {
        "similar.same_threshold",
        "same_threshold",
        "copy_threshold",
        "match_pool",
        "pool",
        "trust.observation",
        "alpha",
        "beta",
        "own_candidate_weight",
        "review_after_days",
        "half_life_days",
        "high_risk_terms",
    }
    keys.update(f"gate.{s}.{f}" for s in _SCOPES for f in _GATE_FIELDS)
    keys.update(f"weight.{c}" for c in _CHANNELS)
    keys.update(f"status_weight.{s}" for s in _STATUSES)
    return keys


def _check_keys(raw: Mapping[str, str]) -> None:
    """Raise ValueError naming the first unknown key."""
    known = _known_keys()
    for key in raw:
        if key not in known:
            raise ValueError(f"unknown ledger config key: {key}")


def _float(key: str, value: str) -> float:
    """Parse a finite float, or raise ValueError naming the key."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key}: not a number: {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{key}: not a finite number: {value!r}")
    return number


def _int(key: str, value: str) -> int:
    """Parse an integer, or raise ValueError naming the key."""
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key}: not an integer: {value!r}") from None


def _bool(key: str, value: str) -> bool:
    """Parse exactly "true" or "false", or raise ValueError naming the key."""
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"{key}: must be 'true' or 'false', got {value!r}")


def _non_negative(key: str, value: float) -> float:
    """Return the value, or raise ValueError if it is negative."""
    if value < 0:
        raise ValueError(f"{key} must be >= 0")
    return value


def _days(key: str, value: float) -> timedelta:
    """A non-negative number of days as a timedelta."""
    return timedelta(days=_non_negative(key, value))
