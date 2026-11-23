"""Tier engine: only what something other than its author has checked becomes
team knowledge. Everything else is handed over as an unchecked note.

A record opens a claim, or joins the claim that already says the same thing.
A claim's standing is read off the evidence attached to it, and standing is a
tier, not a score:

    checked    a test, a document, a reviewer or a person backs it, and nothing
               from outside has turned against it since     -> team knowledge
    unchecked  only its authors and other agents' own reports back it
                                                             -> a note, never knowledge
    refuted    the latest outside verdict went against it   -> not handed over

Rules:

- Promotion needs an outside check. No number of agents agreeing with each
  other promotes a claim: agreement is counted, but it only orders notes.
- The latest outside verdict decides. A passing test followed by a reviewer's
  rejection is refuted; a later passing test restores it. At the same instant
  the rejection wins.
- A verdict (outcome, human_confirm) judges what its refs name, as the
  contract says (docs/contract.md 3-2). Its text is never read as a claim.
- A record's signal says how the statement itself was checked, so a record
  signed by a test, a document or a person is a checked claim.
- Nobody approves their own claim: a review or human verdict from one of the
  claim's authors is ignored, and so is an author reporting their own note as
  working. A test verdict counts whoever reports it; whether the test really
  ran is for the caller to guarantee (the server, a CI hook), not the engine.
- An agent can say that what it wrote replaces what it was handed: it cites
  the handed claim in its record's refs and reports that claim as failing, in
  the same session. That is still only its word, so nothing happens yet. The
  day its own claim passes an outside check, the check counts against the
  claim it replaces, exactly like an outside rejection; if its claim later
  falls, the replacement is withdrawn. So the choice between the two is made
  by the outside check, never by the engine weighing two notes.
- One person repeating themselves is one witness. A record that restates what
  the agent was just handed is the memory talking back, not a witness.
- Nothing fades with time. A claim changes tier only when evidence arrives,
  so tick has nothing to do and every decision is made in observe.

Handoff order, among results that are on topic:

    1. checked knowledge
    2. unchecked notes that two or more people back independently
    3. unchecked notes with one person behind them
    4. unchecked notes someone reported as failing

Inside a tier the shared search decides. Each item ends with one line saying
what backs it. Whoever was handed a claim that has since been refuted gets a
correction, placed right after the best current answer.

Whether two texts state the same thing is the job of a ClaimJudge. The default
judge only merges near-identical wording and never guesses that differently
worded notes agree. A judge that does (a model) can be passed in; then notes
it links count as each other's witnesses. Measured results for such judges
are in the project notes; the default uses none.

Known limits:

- Where nothing has been checked from outside, this engine orders notes the
  way the search does. It does not try to tell which unchecked note is true.
- Tier 2 trusts that people who say the same thing did so independently.
  Recall events (what each session was handed) are what expose copying; a
  caller that never records recalls leaves that check blind.
- Knowledge is never archived for age alone.

Config (optional, strings):
  handoff   "mixed" (knowledge, then notes) or "knowledge" (knowledge only);
            the factory knowledge_only picks "knowledge" without config
"""

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from ..access import audience_sees, owner_for
from ..contract import (
    Caller,
    Decision,
    Demote,
    EngineContext,
    Event,
    Handoff,
    Knowledge,
    Promote,
    Scope,
)

OUTSIDE = ("test", "doc", "review", "human")  # signals that are not the author's own word
ON_TOPIC = 0.4  # share of the question covered, relative to the best hit, that counts as on topic
_POOL = 30  # search candidates read before ordering
_SEP = "\x1f"  # inside state keys; ids may contain ':' or '/'

CHECKED, BACKED, SINGLE, DISPUTED = 0, 1, 2, 3  # handoff tiers, best first

_WORD = re.compile(r"\w+", re.UNICODE)
_HANGUL = re.compile(r"[가-힣]")


def tokens(text: str) -> set[str]:
    """Words, plus Hangul 2-grams so Korean particles do not hide a match."""
    out: set[str] = set()
    for word in _WORD.findall(text.lower()):
        out.add(word)
        if _HANGUL.search(word):
            out.update(word[i : i + 2] for i in range(len(word) - 1))
    return out


class ClaimJudge(Protocol):
    """Decides how two texts relate. The only place wording is interpreted."""

    links: bool  # False: agrees() is never asked, and no candidates are searched for it

    def same(self, a: str, b: str) -> bool:
        """Near-identical: one restates the other (merged into one claim)."""
        ...

    def agrees(self, a: str, b: str) -> bool:
        """Different wording, same fact: the authors witness each other's claim."""
        ...


class RuleJudge:
    """Same claim only when the wording nearly matches; never links notes in
    different words. Merging two different claims pools evidence that does not
    belong together, which is worse than leaving a duplicate."""

    links = False

    def __init__(self, threshold: float = 0.7) -> None:
        self.threshold = threshold

    def same(self, a: str, b: str) -> bool:
        ta, tb = tokens(a), tokens(b)
        if not ta or not tb:
            return False
        return len(ta & tb) / len(ta | tb) >= self.threshold

    def agrees(self, a: str, b: str) -> bool:
        return False


@dataclass(frozen=True)
class Standing:
    """What the evidence says about one claim, with no weights and no clock."""

    checks: tuple[str, ...]  # event ids of outside support, one per witness, oldest first
    refuters: tuple[str, ...]  # event ids of outside refutation, one per witness, oldest first
    refuted: bool  # the latest outside verdict is against it
    witnesses: int  # distinct people who wrote it or reported it working
    failures: int  # distinct people who reported it failing, on their own word
    backing: Mapping[str, int] = field(default_factory=dict)  # signal -> outside witnesses for
    against: Mapping[str, int] = field(default_factory=dict)

    @property
    def checked(self) -> bool:
        return bool(self.checks) and not self.refuted


def _claim_key(cid: str) -> str:
    return f"claim:{cid}"


def _claim(ctx: EngineContext, cid: str) -> dict[str, Any]:
    """The state of a claim the caller knows to exist."""
    claim = ctx.state.get(_claim_key(cid))
    if claim is None:
        raise KeyError(f"no claim {cid!r} in engine state")
    return claim


def _ev_prefix(cid: str) -> str:
    return f"ev:{cid}{_SEP}"


def _of_key(event_id: str) -> str:
    """record id -> the claim it states."""
    return f"of:{event_id}"


def _judged_key(event_id: str) -> str:
    """verdict id -> the claims it judged."""
    return f"judged:{event_id}"


def _link_prefix(cid: str) -> str:
    return f"link:{cid}{_SEP}"


def _seen_key(user: str, cid: str) -> str:
    """This user was handed this claim (any session): they are owed a correction if it falls."""
    return f"seen:{user}{_SEP}{cid}"


def _given_prefix(session: str) -> str:
    return f"given:{session}{_SEP}"


def _cited_prefix(session: str, user: str, cid: str) -> str:
    """Claims this user wrote in this session while citing claim `cid` as handed."""
    return f"cited:{session}{_SEP}{user}{_SEP}{cid}{_SEP}"


def _deny_key(session: str, user: str, cid: str) -> str:
    """This user reported claim `cid` as failing in this session."""
    return f"deny:{session}{_SEP}{user}{_SEP}{cid}"


def _over_prefix(cid: str) -> str:
    """Claims that claim `cid` was declared to replace -> {"at": declared, "ev": check carried over}."""
    return f"over:{cid}{_SEP}"


class TierEngine:
    state_version = "1"

    def __init__(self, config: Mapping[str, str] | None = None, judge: ClaimJudge | None = None) -> None:
        config = config or {}
        self.mode = config.get("handoff", "mixed")
        if self.mode not in ("mixed", "knowledge"):
            raise ValueError(f"handoff must be 'mixed' or 'knowledge', not {self.mode!r}")
        self.judge = judge or RuleJudge()

    # ── observe ────────────────────────────────────────────────────────────

    def observe(self, event: Event, ctx: EngineContext) -> Sequence[Decision]:
        if event.kind == "record":
            return self._on_record(event, ctx)
        if event.kind == "recall":
            for cid in event.refs:
                if ctx.state.get(_claim_key(cid)) is not None:
                    ctx.state.put(_given_prefix(event.session) + cid, 1)
                    ctx.state.put(_seen_key(event.user, cid), event.at.isoformat())
            return []
        if event.kind in ("outcome", "human_confirm"):
            return self._on_verdict(event, ctx)
        return []  # commit: carries no claim and judges none

    def _on_record(self, event: Event, ctx: EngineContext) -> list[Decision]:
        if not event.text.strip():
            return []
        echoed = self._echo_of(event, ctx)
        if echoed is not None:
            # The agent repeating what it was handed: tie it to the claim, add no witness.
            ctx.state.put(_of_key(event.id), echoed)
            return self._cite(event, echoed, ctx)
        cid = self._same_claim(event, ctx) or self._open_claim(event, ctx)
        ctx.state.put(_of_key(event.id), cid)
        claim = _claim(ctx, cid)
        if event.user not in claim["authors"]:
            claim["authors"].append(event.user)
            ctx.state.put(_claim_key(cid), claim)
        # A record states something; its signal says how the statement was checked.
        # A test record whose outcome is failure still states a true thing: "this fails".
        self._attach(ctx, cid, event, +1, event.signal or "self")
        self._cite(event, cid, ctx)
        touched = [cid, *self._link(event, cid, ctx)]
        return [d for c in dict.fromkeys(touched) for d in self._settle(c, event.at, ctx)]

    def _cite(self, event: Event, cid: str, ctx: EngineContext) -> list[Decision]:
        """The record states claim `cid` and cites other claims as handed
        before it was written. Only an author of `cid` can tie it to them."""
        if event.user not in _claim(ctx, cid)["authors"]:
            return []
        declared = False
        for old in self._resolve(event.refs, ctx):
            if old == cid:
                continue
            ctx.state.put(_cited_prefix(event.session, event.user, old) + cid, 1)
            if ctx.state.get(_deny_key(event.session, event.user, old)) is not None:
                self._declare(cid, old, event.at, ctx)
                declared = True
        return self._settle(cid, event.at, ctx) if declared else []

    def _on_verdict(self, event: Event, ctx: EngineContext) -> list[Decision]:
        cids = self._resolve(event.refs, ctx)
        ctx.state.put(_judged_key(event.id), cids)
        if event.outcome not in ("success", "failure"):
            return []
        signal = event.signal or ("human" if event.kind == "human_confirm" else "self")
        decisions: list[Decision] = []
        for cid in cids:
            own = event.user in _claim(ctx, cid)["authors"]
            if event.outcome == "success" and own and signal != "test":
                continue  # approving your own claim, or reporting your own note as working
            self._attach(ctx, cid, event, +1 if event.outcome == "success" else -1, signal)
            decisions += self._settle(cid, event.at, ctx)
            if event.outcome == "failure":
                ctx.state.put(_deny_key(event.session, event.user, cid), event.at.isoformat())
                prefix = _cited_prefix(event.session, event.user, cid)
                for key, _ in list(ctx.state.scan(prefix)):
                    new = key[len(prefix):]
                    self._declare(new, cid, event.at, ctx)
                    decisions += self._settle(new, event.at, ctx)
        return decisions

    def _declare(self, new: str, old: str, at: datetime, ctx: EngineContext) -> None:
        """Note that the author of `new` says it replaces `old`. Takes effect in _carry."""
        if ctx.state.get(_over_prefix(new) + old) is None:
            ctx.state.put(_over_prefix(new) + old, {"at": at.isoformat(), "ev": None})

    def _attach(self, ctx: EngineContext, cid: str, event: Event, polarity: int, signal: str) -> None:
        claim = _claim(ctx, cid)
        if not audience_sees(event, claim["scope"], claim["owner"]):
            return  # the claim's audience could not check it, so it does not count
        ctx.state.put(_ev_prefix(cid) + event.id, {
            "who": _witness(event, signal), "pol": polarity, "signal": signal, "at": event.at.isoformat(),
        })

    def _link(self, event: Event, cid: str, ctx: EngineContext) -> list[str]:
        """Claims by other people that the judge takes to state the same fact
        in other words. They are not merged: each keeps its own evidence, and
        the link only makes their authors witnesses of each other."""
        if not self.judge.links:
            return []
        caller = Caller(user=event.user, team=event.team, session=event.session, at=event.at)
        mine = _claim(ctx, cid)
        touched = []
        for other, _ in ctx.events.search(event.text, caller, event.scope, k=_POOL, kinds=("record",)):
            if other.user == event.user or other.id == event.id:
                continue
            ocid = ctx.state.get(_of_key(other.id))
            if ocid is None or ocid == cid or ctx.state.get(_link_prefix(cid) + ocid):
                continue
            theirs = _claim(ctx, ocid)
            if (theirs["scope"], theirs["owner"], theirs["conditions"]) != (
                mine["scope"], mine["owner"], mine["conditions"]
            ):
                continue
            if not self.judge.agrees(event.text, other.text):
                continue
            ctx.state.put(_link_prefix(cid) + ocid, 1)
            ctx.state.put(_link_prefix(ocid) + cid, 1)
            touched.append(ocid)
        return touched

    # ── tick ───────────────────────────────────────────────────────────────

    def tick(self, now: datetime, ctx: EngineContext) -> Sequence[Decision]:
        return []  # nothing depends on time

    # ── handoff ────────────────────────────────────────────────────────────

    def handoff(
        self, query: str, caller: Caller, scope: Scope, ctx: EngineContext, k: int = 6
    ) -> list[Handoff]:
        context = dict(caller.context)
        if caller.repo:
            context.setdefault("repo", caller.repo)

        # 1. What the shared search finds, and how much of the question each hit
        #    covers (0..1, relative to the best). A verdict that matches the
        #    query makes the claims it judged relevant.
        k_hits = ctx.knowledge.search(query, caller, scope, k=max(k, _POOL))
        e_hits = ctx.events.search(query, caller, scope, k=max(k, _POOL),
                                   kinds=("record", "outcome", "human_confirm"))
        k_rel, e_rel = _relevance(query, [i.body for i, _ in k_hits], [e.text for e, _ in e_hits])
        rel: dict[str, float] = {}
        # Where the search put it: its best position among the events, where notes
        # and knowledge compete; knowledge found only as knowledge comes after those.
        place: dict[str, int] = {}
        items: dict[str, Knowledge] = {}
        for at, ((item, _), r) in enumerate(zip(k_hits, k_rel)):
            items[item.id] = item
            rel[item.id] = max(rel.get(item.id, 0.0), r)
            place[item.id] = len(e_hits) + at
        for at, ((event, _), r) in enumerate(zip(e_hits, e_rel)):
            if event.kind == "record":
                cid = ctx.state.get(_of_key(event.id))
                cids = [] if cid is None else [cid]
            else:
                cids = ctx.state.get(_judged_key(event.id)) or []
            for cid in cids:
                rel[cid] = max(rel.get(cid, 0.0), r)
                place[cid] = min(place.get(cid, at), at)

        # 2. Claims the judge linked share relevance: a note in other words is as
        #    on topic as the note the search matched.
        for cid, r in list(rel.items()):
            for key, _ in ctx.state.scan(_link_prefix(cid)):
                other = key[len(_link_prefix(cid)):]
                rel[other] = max(rel.get(other, 0.0), r)
                place[other] = min(place.get(other, place[cid]), place[cid])

        # 3. Tier each claim; collect corrections owed to this user.
        ranked: list[tuple[tuple[float, ...], str, Handoff]] = []
        owed: list[tuple[float, str, Handoff]] = []
        for cid, r in rel.items():
            claim = ctx.state.get(_claim_key(cid))
            if claim is None:  # knowledge this engine's state does not know (state was reset)
                item = items.get(cid)
                if item is not None and (miss := _unmet(item.conditions, context)) is not None:
                    ranked.append((_rank(CHECKED, miss, r, place[cid]), cid, Handoff(
                        id=cid, content=item.body, sources=item.sources, status="knowledge",
                        conditions=item.conditions)))
                continue
            miss = _unmet(claim["conditions"], context)
            if miss is None:
                continue  # holds somewhere else than where the question was asked
            standing = self._standing(cid, ctx)
            if claim["status"] == "knowledge":
                sources = tuple(claim["sources"])
                ranked.append((_rank(CHECKED, miss, r, place[cid]), cid, Handoff(
                    id=cid, content=_with_basis(claim["text"], standing, checked=True), sources=sources,
                    status="knowledge", conditions=claim["conditions"])))
            elif claim["status"] == "memory":
                if self.mode != "mixed":
                    continue
                if standing.failures:
                    tier = DISPUTED
                else:
                    tier = BACKED if standing.witnesses >= 2 else SINGLE
                ranked.append((_rank(tier, miss, r, place[cid]), cid, Handoff(
                    id=cid, content=_with_basis(claim["text"], standing, checked=False),
                    sources=(claim["first"],), status="memory", conditions=claim["conditions"])))
            elif standing.refuters and ctx.state.get(_seen_key(caller.user, cid)) and r >= ON_TOPIC:
                owed.append((-r, cid, Handoff(
                    id=f"x-{cid}", content=_retraction(claim["text"], standing, ctx, self._replacers(cid, standing, ctx)),
                    sources=standing.refuters, status="invalidated", conditions=claim["conditions"])))

        # 4. Tier orders what is on topic; a correction follows the best current answer.
        best = [h for _, _, h in sorted(ranked, key=lambda t: (t[0], t[1]))]
        corrections = [h for _, _, h in sorted(owed, key=lambda t: (t[0], t[1]))]
        return (best[:1] + corrections + best[1:])[:k]

    # ── claims and evidence ────────────────────────────────────────────────

    def _open_claim(self, event: Event, ctx: EngineContext) -> str:
        cid = f"c-{event.id}"
        ctx.state.put(_claim_key(cid), {
            "id": cid, "text": event.text, "scope": event.scope, "owner": owner_for(event),
            "conditions": dict(event.conditions), "first": event.id, "authors": [],
            "status": "memory", "sources": [],
        })
        return cid

    def _same_claim(self, event: Event, ctx: EngineContext) -> str | None:
        """An existing claim with the same audience that says the same thing."""
        caller = Caller(user=event.user, team=event.team, session=event.session, at=event.at)
        owner = owner_for(event)
        for other, _ in ctx.events.search(event.text, caller, event.scope, k=10, kinds=("record",)):
            if other.id == event.id:
                continue
            cid = ctx.state.get(_of_key(other.id))
            if cid is None:
                continue
            claim = _claim(ctx, cid)
            if (claim["scope"], claim["owner"]) != (event.scope, owner):
                continue
            if dict(event.conditions) != claim["conditions"]:
                continue  # same words, different place: a different claim
            if self.judge.same(event.text, claim["text"]):
                return cid
        return None

    def _echo_of(self, event: Event, ctx: EngineContext) -> str | None:
        """The claim this record restates, if the agent was handed it earlier."""
        candidates = list(self._resolve(event.refs, ctx))
        for key, _ in ctx.state.scan(_given_prefix(event.session)):
            candidates.append(key[len(_given_prefix(event.session)):])
        for cid in dict.fromkeys(candidates):
            claim = ctx.state.get(_claim_key(cid))
            if claim and self.judge.same(event.text, claim["text"]):
                return cid
        return None

    def _replacers(self, cid: str, standing: Standing, ctx: EngineContext) -> list[str]:
        """Texts of the checked claims whose check stands against this one."""
        out = []
        for eid in standing.refuters:
            by = (ctx.state.get(_ev_prefix(cid) + eid) or {}).get("by")
            if by is not None:
                out.append(_claim(ctx, by)["text"].strip())
        return out

    def _resolve(self, refs: Iterable[str], ctx: EngineContext) -> list[str]:
        """Refs name a claim (handoff or knowledge id) or a record inside one."""
        out = []
        for ref in refs:
            if ctx.state.get(_claim_key(ref)) is not None:
                out.append(ref)
            elif (cid := ctx.state.get(_of_key(ref))) is not None:
                out.append(cid)
        return list(dict.fromkeys(out))

    def _standing(self, cid: str, ctx: EngineContext) -> Standing:
        prefix = _ev_prefix(cid)
        latest: dict[tuple[str, int], tuple[datetime, str, str]] = {}  # (who, pol) -> (at, event id, signal)
        people: dict[int, set[str]] = {+1: set(), -1: set()}
        for key, ev in ctx.state.scan(prefix):
            if ev["signal"] not in OUTSIDE:
                people[ev["pol"]].add(ev["who"])
                continue
            entry = (datetime.fromisoformat(ev["at"]), key[len(prefix):], ev["signal"])
            slot = (ev["who"], ev["pol"])
            if slot not in latest or entry[:2] > latest[slot][:2]:
                latest[slot] = entry

        def side(pol: int) -> list[tuple[datetime, str, str]]:
            return sorted(entry for (_, p), entry in latest.items() if p == pol)

        def kinds(entries: list[tuple[datetime, str, str]]) -> dict[str, int]:
            out: dict[str, int] = {}
            for _, _, signal in entries:
                out[signal] = out.get(signal, 0) + 1
            return dict(sorted(out.items()))

        support, refute = side(+1), side(-1)
        refuted = bool(refute) and (not support or refute[-1][0] >= support[-1][0])
        witnesses = set(people[+1])
        for key, _ in ctx.state.scan(_link_prefix(cid)):
            linked = _claim(ctx, key[len(_link_prefix(cid)):])
            witnesses.update(f"user:{u}" for u in linked["authors"])
        return Standing(
            checks=tuple(eid for _, eid, _ in support),
            refuters=tuple(eid for _, eid, _ in refute),
            refuted=refuted,
            witnesses=len(witnesses),
            failures=len(people[-1]),
            backing=kinds(support),
            against=kinds(refute),
        )

    def _settle(self, cid: str, now: datetime, ctx: EngineContext,
                seen: frozenset[str] = frozenset()) -> list[Decision]:
        """Bring the knowledge base in line with the claim's standing, then
        with the standing of whatever this claim was declared to replace."""
        decisions = self._settle_one(cid, now, ctx)
        return decisions + self._carry(cid, now, ctx, seen | {cid})

    def _carry(self, cid: str, now: datetime, ctx: EngineContext, seen: frozenset[str]) -> list[Decision]:
        """While this claim is checked, its latest check stands against each
        claim it was declared to replace; once it is not, that is withdrawn."""
        prefix = _over_prefix(cid)
        standing = self._standing(cid, ctx)
        decisions: list[Decision] = []
        for key, link in list(ctx.state.scan(prefix)):
            old = key[len(prefix):]
            want = standing.checks[-1] if standing.checked else None
            if old in seen and want is not None:
                continue  # two claims replacing each other: the one settled first stands
            if want is not None and want != link["ev"] and ctx.state.get(_ev_prefix(old) + want) is not None:
                want = link["ev"]  # that check already judged the old claim itself; leave it be
            if want == link["ev"]:
                continue
            if link["ev"] is not None:
                ctx.state.delete(_ev_prefix(old) + link["ev"])
            carried = None
            if want is not None:
                check: dict[str, Any] = ctx.state.get(_ev_prefix(cid) + want) or {}
                target = _claim(ctx, old)
                [event] = ctx.events.get([want])
                if audience_sees(event, target["scope"], target["owner"]):
                    ctx.state.put(_ev_prefix(old) + want, {
                        "who": f"replaced:{cid}", "pol": -1, "signal": check["signal"],
                        "at": max(check["at"], link["at"], key=datetime.fromisoformat), "by": cid,
                    })
                    carried = want
            ctx.state.put(key, {"at": link["at"], "ev": carried})
            target = _claim(ctx, old)
            if target["status"] == "dropped" and not self._standing(old, ctx).refuted:
                target["status"] = "memory"  # the replacement fell; back to what its own evidence says
                ctx.state.put(_claim_key(old), target)
            if old not in seen:
                decisions += self._settle(old, now, ctx, seen)
        return decisions

    def _settle_one(self, cid: str, now: datetime, ctx: EngineContext) -> list[Decision]:
        claim = _claim(ctx, cid)
        standing = self._standing(cid, ctx)
        if standing.checked:
            sources = list(dict.fromkeys((claim["first"], *standing.checks)))
            if claim["status"] == "knowledge" and sources == claim["sources"]:
                return []
            claim["status"], claim["sources"] = "knowledge", sources  # new, restored, or newly cited
            ctx.state.put(_claim_key(cid), claim)
            return [Promote(_item(claim))]
        if standing.refuted and claim["status"] != "dropped":
            was_knowledge = claim["status"] == "knowledge"
            claim["status"] = "dropped"
            ctx.state.put(_claim_key(cid), claim)
            if was_knowledge:
                return [Demote(cid, "an outside check turned against it", standing.refuters,
                               why="invalid", since=now)]
        return []


def knowledge_only(config: Mapping[str, str] | None = None) -> TierEngine:
    """Factory for the "knowledge" handoff mode, for callers that pass no config
    (the server builds engines with {}): cortex_mem.engines.tiers:knowledge_only"""
    return TierEngine({**(config or {}), "handoff": "knowledge"})


def _witness(event: Event, signal: str) -> str:
    """What makes two pieces of evidence the same witness: one test run or
    document revision, or one person."""
    if signal in ("test", "doc"):
        if event.repo or event.rev:
            return f"{signal}:{event.repo}@{event.rev}"
        return f"{signal}:{event.id}"
    return f"{signal if signal in OUTSIDE else 'user'}:{event.user}"


def _item(claim: Mapping[str, Any]) -> Knowledge:
    text = claim["text"]
    title = next((line.strip() for line in text.splitlines() if line.strip()), claim["id"])[:80]
    return Knowledge(
        id=claim["id"], kind="knowledge", title=title, body=text, scope=claim["scope"],
        owner=claim["owner"], sources=tuple(claim["sources"]), conditions=claim["conditions"],
    )


_SIGNAL_NAMES = {"test": "테스트", "doc": "문서", "review": "리뷰", "human": "사람 확인"}


def _named(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{_SIGNAL_NAMES.get(s, s)} {n}" for s, n in counts.items())


def _with_basis(text: str, standing: Standing, checked: bool) -> str:
    """The note, then one line saying what backs it, so the reader can weigh it."""
    if checked:
        parts = [f"확인된 지식 · 확인: {_named(standing.backing)}"]
    else:
        parts = [f"미확인 기록 · 작성·사용 보고 {standing.witnesses}명 · 외부 확인 없음"]
    if standing.failures:
        parts.append(f"실패 보고 {standing.failures}명")
    return f"{text}\n\n[근거] " + " · ".join(parts)


def _retraction(text: str, standing: Standing, ctx: EngineContext, replaced_by: Sequence[str] = ()) -> str:
    why = [e.text.strip().splitlines()[0][:200] for e in ctx.events.get(standing.refuters)
           if e.text.strip() and not e.text_redacted]
    if replaced_by:
        lines = ["[철회] 예전에 건넨 아래 내용은 이후 외부 확인을 받은 다른 내용으로 대체되었습니다. 따르지 마세요.",
                 f"이전 내용: {text.strip()}"]
        lines += [f"대체한 내용: {t}" for t in replaced_by[:2]]
        lines.append(f"확인 근거: {_named(standing.against)}")
    else:
        lines = ["[철회] 예전에 건넨 아래 내용은 이후 외부 확인에서 틀린 것으로 판정되었습니다. 따르지 마세요.",
                 f"이전 내용: {text.strip()}",
                 f"반박 근거: {_named(standing.against)}"]
    lines += [f"- {w}" for w in why[:3]]
    return "\n".join(lines)


def _rank(tier: int, unmet: int, rel: float, place: int) -> tuple[int, ...]:
    """Sort key, smallest first. A tier above a single unchecked note counts
    only for what is on topic; then how well the conditions fit where the
    question was asked; then the order the search gave. When nothing has been
    checked or backed, that leaves the search's own order."""
    return (tier if rel >= ON_TOPIC else max(tier, SINGLE), unmet, place)


def _relevance(query: str, *lists: Sequence[str]) -> list[list[float]]:
    """For each text, the share of the question's tokens it contains, scaled
    so the best text of all lists is 1. Counted here rather than taken from
    the stores: their scores depend on the ranking formula and cannot be
    compared between two result lists, so a store decides which texts are
    candidates and in what order equals stand, not what is on topic. When
    nothing shares a token with the question, order within each list stands."""
    asked = tokens(query)
    shared = [[len(asked & tokens(text)) for text in texts] for texts in lists]
    top = max((n for counts in shared for n in counts), default=0)
    if top == 0:
        return [[1.0 / (i + 1) for i in range(len(counts))] for counts in shared]
    return [[n / top for n in counts] for counts in shared]


def _unmet(conditions: Mapping[str, str], context: Mapping[str, str]) -> int | None:
    """None when a condition contradicts where the question was asked;
    otherwise how many conditions the question does not mention."""
    unmet = 0
    for key, value in conditions.items():
        if key in context:
            if context[key] != value:
                return None
        else:
            unmet += 1
    return unmet
