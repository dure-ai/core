from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from cortex_mem.contract import Event
from cortex_mem.engines.ledger.config import LedgerConfig
from cortex_mem.engines.ledger.evidence import (
    VERIFIED,
    channel_of,
    in_scope,
    independence_key,
    tally,
)
from cortex_mem.engines.ledger.model import Claim, Evidence

KST = timezone(timedelta(hours=9))
T0 = datetime(2026, 11, 19, 10, 0, tzinfo=KST)
CFG = LedgerConfig.from_mapping(None)


def ev(kind="record", *, scope="team", user="u1", team="t1", session="s1",
       signal=None, outcome=None, eid="e1") -> Event:
    return Event(id=eid, kind=kind, at=T0, user=user, team=team, session=session,
                 scope=scope, signal=signal, outcome=outcome)


def claim(scope="team", owner="t1") -> Claim:
    return Claim(id="c1", scope=scope, owner=owner, conditions={}, canonical="x",
                 status="candidate", first_at=T0, last_support_at=T0,
                 invalid_since=None, promoted=False, use_success=0, use_failure=0)


def e(key, channel="observation", stance="support", derived=None, days=0, eid=None):
    return Evidence(claim_id="c1", event_id=eid or f"{key}-{channel}-{stance}-{days}",
                    stance=stance, channel=channel, key=key, user="u",
                    at=T0 + timedelta(days=days), derived=derived)


def test_verified_set():
    assert VERIFIED == frozenset({"review", "test", "doc", "human"})


def test_in_scope_team_claim():
    c = claim("team", "t1")
    assert in_scope(ev(scope="team", team="t1"), c)
    assert in_scope(ev(scope="org", team="t2"), c)
    assert not in_scope(ev(scope="personal", team="t1"), c)
    assert not in_scope(ev(scope="team", team="t2"), c)


def test_in_scope_personal_claim():
    c = claim("personal", "u1")
    assert in_scope(ev(scope="personal", user="u1"), c)
    assert not in_scope(ev(scope="personal", user="u2"), c)


@pytest.mark.parametrize(
    ("kind", "signal", "want"),
    [
        ("record", "self", "observation"),
        ("record", None, "observation"),
        ("record", "test", "test"),
        ("record", "human", "human"),
        ("outcome", "review", "review"),
        ("outcome", "doc", "doc"),
        ("outcome", "self", "observation"),
        ("outcome", None, "observation"),
        ("human_confirm", None, "human"),
        ("human_confirm", "self", "human"),
        ("human_confirm", "test", "human"),
    ],
)
def test_channel_of(kind, signal, want):
    assert channel_of(ev(kind, signal=signal)) == want


def test_independence_key():
    e1 = ev(user="u1", session="s9")
    assert independence_key("personal", e1) == "s:s9"
    assert independence_key("team", e1) == "u:u1"


def test_same_key_repeats_are_one_vote():
    t = tally([e("u:a"), e("u:a"), e("u:a")], CFG)
    assert t.S == pytest.approx(0.7)
    assert t.support_keys == 1


def test_strongest_channel_per_key():
    t = tally([e("u:a"), e("u:a", "test")], CFG)
    assert t.S == pytest.approx(2.0)
    assert t.by_channel == {"test": 1}
    assert t.verified_keys == 1


def test_two_people():
    t = tally([e("u:a"), e("u:b")], CFG)
    assert t.S == pytest.approx(1.4)
    assert t.support_keys == 2
    assert t.verified_keys == 0
    assert t.by_channel == {"observation": 2}
    assert t.confidence == pytest.approx((1.4 + 1) / (1.4 + 0 + 2))


def test_derived_not_counted():
    t = tally([e("u:a", derived="copy", days=-5), e("u:b", derived="retrieved")], CFG)
    assert t.S == 0
    assert t.support_keys == 0
    assert t.first_support is None
    assert t.last_support is None


def test_support_dates():
    t = tally([e("u:a", days=3), e("u:b", days=1), e("u:c", derived="copy", days=-9)], CFG)
    assert t.first_support == T0 + timedelta(days=1)
    assert t.last_support == T0 + timedelta(days=3)


def test_verified_contradiction():
    t = tally([e("u:a", "human", "contradict")], CFG)
    assert t.C == pytest.approx(3.0)
    assert t.contra_keys == 1
    assert t.verified_contra is True


def test_observation_contradiction_not_verified():
    t = tally([e("u:a", "observation", "contradict")], CFG)
    assert t.C == pytest.approx(0.7)
    assert t.verified_contra is False


def test_same_key_support_and_contra_tallied_separately():
    t = tally([e("u:a"), e("u:a", stance="contradict")], CFG)
    assert t.support_keys == 1
    assert t.contra_keys == 1


def test_empty():
    t = tally([], CFG)
    assert t.S == 0 and t.C == 0
    assert t.confidence == 0.5
    assert t.first_support is None and t.last_support is None
    assert t.by_channel == {}
