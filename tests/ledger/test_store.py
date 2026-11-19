from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import Any

from cortex_mem.engines.ledger.model import Claim, Evidence, Memory
from cortex_mem.engines.ledger.store import LedgerStore

KST = timezone(timedelta(hours=9))
T0 = datetime(2026, 11, 19, 10, 0, tzinfo=KST)


class FakeState:
    def __init__(self) -> None:
        self.data: dict[str, Any] = {}

    def get(self, key: str) -> Any | None:
        v = self.data.get(key)
        return None if v is None else json.loads(json.dumps(v))

    def put(self, key: str, value: Any) -> None:
        self.data[key] = json.loads(json.dumps(value))

    def delete(self, key: str) -> None:
        self.data.pop(key, None)

    def scan(self, prefix: str = "") -> Iterator[tuple[str, Any]]:
        for k in sorted(self.data):
            if k.startswith(prefix):
                yield k, json.loads(json.dumps(self.data[k]))


def mem(group: str | None = None) -> Memory:
    return Memory(
        id="e1", claim_id="c-e1", user="u", team="t", session="s", at=T0, group=group
    )


def claim(cid: str = "c-e1") -> Claim:
    return Claim(
        id=cid,
        scope="team",
        owner="t",
        conditions={"os": "mac"},
        canonical="e1",
        status="candidate",
        first_at=T0,
        last_support_at=T0,
        invalid_since=T0 + timedelta(days=1),
        promoted=False,
        use_success=1,
        use_failure=2,
        report={"x": [1, 2]},
    )


def ev(cid: str, eid: str, at: datetime = T0) -> Evidence:
    return Evidence(
        claim_id=cid,
        event_id=eid,
        stance="support",
        channel="review",
        key="k",
        user="u",
        at=at,
        derived=None,
    )


def test_roundtrip_memory_claim_evidence() -> None:
    for m in (mem(None), mem("g1")):
        assert Memory.from_json(json.loads(json.dumps(m.to_json()))) == m
    c = claim()
    assert Claim.from_json(json.loads(json.dumps(c.to_json()))) == c
    c2 = claim()
    c2.invalid_since = None
    assert Claim.from_json(json.loads(json.dumps(c2.to_json()))) == c2
    for d in (None, "copy"):
        e = Evidence("c", "e", "contradict", "test", "k", "u", T0, d)
        assert Evidence.from_json(json.loads(json.dumps(e.to_json()))) == e


def test_timezone_preserved() -> None:
    back = Memory.from_json(json.loads(json.dumps(mem().to_json())))
    assert back.at == T0
    assert back.at.utcoffset() == timedelta(hours=9)


def test_memory_claim_crud() -> None:
    s = LedgerStore(FakeState())
    assert s.memory("e1") is None and s.claim("c-e1") is None
    s.put_memory(mem())
    s.put_claim(claim("c-b"))
    s.put_claim(claim("c-a"))
    assert s.memory("e1") == mem()
    assert s.claim("c-a") == claim("c-a")
    assert [c.id for c in s.claims()] == ["c-a", "c-b"]


def test_evidence_order_and_overwrite() -> None:
    s = LedgerStore(FakeState())
    s.put_evidence(ev("c-1", "e2", T0 + timedelta(hours=1)))
    s.put_evidence(ev("c-1", "e9", T0))
    s.put_evidence(ev("c-1", "e1", T0))
    s.put_evidence(ev("c-1", "e1", T0))
    s.put_evidence(ev("c-10", "e5"))
    assert [(e.event_id) for e in s.evidence("c-1")] == ["e1", "e9", "e2"]
    assert [e.event_id for e in s.evidence("c-10")] == ["e5"]


def test_group_prefix_not_mixed() -> None:
    s = LedgerStore(FakeState())
    s.add_to_group("e1", "c-b")
    s.add_to_group("e1", "c-a")
    s.add_to_group("e1", "c-a")
    s.add_to_group("e10", "c-z")
    assert s.claims_in_group("e1") == ["c-a", "c-b"]
    assert s.claims_in_group("e10") == ["c-z"]


def test_idempotent_writes() -> None:
    s = LedgerStore(FakeState())
    s.mark_seen("s", "c1", T0)
    s.mark_seen("s", "c1", T0 + timedelta(hours=5))
    assert s.seen("s", "c1") == T0
    assert s.seen("s", "c2") is None
    s.index_signature("team", "t", "sig", "c-1")
    s.index_signature("team", "t", "sig", "c-2")
    assert s.claim_by_signature("team", "t", "sig") == "c-1"
    assert s.claim_by_signature("team", "t", "other") is None
    s.index_signature("team", "t", "sig", "c-3", replace=True)
    assert s.claim_by_signature("team", "t", "sig") == "c-3"


def test_skip_counts() -> None:
    s = LedgerStore(FakeState())
    assert s.skipped() == {}
    s.skip("unknown_ref")
    s.skip("unknown_ref")
    assert s.skipped() == {"unknown_ref": 2}


def test_key_rules() -> None:
    st = FakeState()
    s = LedgerStore(st)
    s.put_memory(mem())
    s.put_claim(claim())
    s.put_evidence(ev("c-e1", "e1"))
    s.mark_seen("sess", "c-e1", T0)
    s.add_to_group("g", "c-e1")
    s.index_signature("team", "t", "sg", "c-e1")
    s.skip("r")
    assert set(st.data) == {
        "mem:e1",
        "claim:c-e1",
        "ev:c-e1:e1",
        "seen:sess:c-e1",
        "grp:g:c-e1",
        "sig:team:t:sg",
        "ledger:skipped",
    }
    assert st.data["grp:g:c-e1"] is True
    assert st.data["sig:team:t:sg"] == "c-e1"


def test_colon_in_claim_id_does_not_collide_evidence() -> None:
    s = LedgerStore(FakeState())
    s.put_evidence(ev("c-a", "e1"))
    s.put_evidence(ev("c-a:b", "e2"))
    assert [e.event_id for e in s.evidence("c-a")] == ["e1"]
    assert [e.event_id for e in s.evidence("c-a:b")] == ["e2"]


def test_group_ids_with_special_chars() -> None:
    s = LedgerStore(FakeState())
    s.add_to_group("g", "c-1")
    s.add_to_group("g:x", "c-2")
    s.add_to_group("g", "c:y")
    s.add_to_group("g", "c%z")
    s.add_to_group("g", "c%3Ay")
    assert s.claims_in_group("g") == sorted(["c-1", "c:y", "c%z", "c%3Ay"])
    assert s.claims_in_group("g:x") == ["c-2"]


def test_seen_components_stay_distinct() -> None:
    s = LedgerStore(FakeState())
    s.mark_seen("s", "c:1", T0)
    assert s.seen("s:c", "1") is None
    assert s.seen("s", "c:1") == T0
    s.mark_seen("s:c", "1", T0 + timedelta(hours=1))
    assert s.seen("s", "c:1") == T0
    assert s.seen("s:c", "1") == T0 + timedelta(hours=1)


def test_signature_components_stay_distinct() -> None:
    s = LedgerStore(FakeState())
    s.index_signature("team", "t1:x", "sig", "c-1")
    assert s.claim_by_signature("team", "t1", "x:sig") is None
    s.index_signature("team", "t1", "x:sig", "c-2")
    assert s.claim_by_signature("team", "t1:x", "sig") == "c-1"
    assert s.claim_by_signature("team", "t1", "x:sig") == "c-2"


def test_percent_ids_roundtrip() -> None:
    s = LedgerStore(FakeState())
    c = claim("c-%41")
    s.put_claim(c)
    s.put_evidence(ev("c-%41", "e%3A1"))
    s.put_memory(Memory("e%1", "c-%41", "u", "t", "s", T0, None))
    assert s.claim("c-%41") == c
    assert [x.event_id for x in s.evidence("c-%41")] == ["e%3A1"]
    assert s.memory("e%1") is not None
    assert [x.id for x in s.claims()] == ["c-%41"]


def test_judged_round_trip_and_components_do_not_collide():
    st = LedgerStore(FakeState())
    assert not st.judged("c-1", "e1")
    st.mark_judged("c-1", "e1")
    assert st.judged("c-1", "e1")
    assert not st.judged("c-1", "e2") and not st.judged("c-2", "e1")
    st.mark_judged("a:b", "c")
    assert not st.judged("a", "b:c")
    st.mark_judged("a%3Ab", "d")
    assert not st.judged("a:b", "d")
