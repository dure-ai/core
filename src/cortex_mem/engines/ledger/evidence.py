"""Evidence scope, channel, independence and tally (spec section 8).

Pure functions: no clock, no randomness, no state access.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType

from cortex_mem.access import audience_sees
from cortex_mem.contract import Event, Scope
from cortex_mem.engines.ledger.config import LedgerConfig
from cortex_mem.engines.ledger.model import Channel, Claim, Evidence

VERIFIED: frozenset[str] = frozenset({"review", "test", "doc", "human"})

# Tie-break order when two channels of one key are equally strong.
_CHANNEL_ORDER: tuple[Channel, ...] = ("human", "test", "doc", "review", "observation")


def in_scope(event: Event, claim: Claim) -> bool:
    """Whether the event may serve as evidence for the claim."""
    return audience_sees(event, claim.scope, claim.owner)


def channel_of(event: Event) -> Channel:
    """The evidence channel of an event."""
    if event.kind == "human_confirm" or event.signal == "human":
        return "human"
    if event.signal in ("test", "review", "doc"):
        return event.signal
    return "observation"


def independence_key(scope: Scope, event: Event) -> str:
    """Events with the same key count as one vote."""
    if scope == "personal":
        return f"s:{event.session}"
    return f"u:{event.user}"


@dataclass(frozen=True)
class Tally:
    """A claim's evidence counted per independent source (derived rows excluded).

    - `S`, `C`: summed strength of support and contradiction, one source
      counting once at its strongest channel.
    - `support_keys`, `verified_keys`, `contra_keys`: independent sources
      that support, support through a verified channel, and contradict.
    - `verified_contra`: whether any contradiction came through a verified
      channel.
    - `first_support`, `last_support`: time span of the support.
    - `by_channel`: support sources by their strongest channel.
    - `confidence`: (S + alpha) / (S + C + alpha + beta).
    """

    S: float
    C: float
    support_keys: int
    verified_keys: int
    contra_keys: int
    verified_contra: bool
    first_support: datetime | None
    last_support: datetime | None
    by_channel: Mapping[str, int]
    confidence: float


def tally(evidence: Sequence[Evidence], cfg: LedgerConfig) -> Tally:
    """Tally non-derived evidence into support, contradiction and confidence."""

    def strength(ch: Channel) -> float:
        trust = cfg.trust_observation if ch == "observation" else 1.0
        return cfg.weights[ch] * trust

    def best(items: list[Evidence]) -> Channel:
        return max(
            (x.channel for x in items),
            key=lambda ch: (strength(ch), -_CHANNEL_ORDER.index(ch)),
        )

    def group(stance: str) -> dict[str, list[Evidence]]:
        keys: dict[str, list[Evidence]] = {}
        for x in evidence:
            if x.derived is None and x.stance == stance:
                keys.setdefault(x.key, []).append(x)
        return keys

    support = group("support")
    contra = group("contradict")

    s_total = 0.0
    verified = 0
    by_channel: dict[str, int] = {}
    for key in sorted(support):
        ch = best(support[key])
        s_total += strength(ch)
        by_channel[ch] = by_channel.get(ch, 0) + 1
        if any(x.channel in VERIFIED for x in support[key]):
            verified += 1

    c_total = sum(strength(best(contra[key])) for key in sorted(contra))
    verified_contra = any(x.channel in VERIFIED for xs in contra.values() for x in xs)

    times = [x.at for xs in support.values() for x in xs]
    return Tally(
        S=s_total,
        C=c_total,
        support_keys=len(support),
        verified_keys=verified,
        contra_keys=len(contra),
        verified_contra=verified_contra,
        first_support=min(times) if times else None,
        last_support=max(times) if times else None,
        by_channel=MappingProxyType(by_channel),
        confidence=(s_total + cfg.alpha) / (s_total + c_total + cfg.alpha + cfg.beta),
    )
