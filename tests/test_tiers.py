from datetime import datetime, timedelta, timezone

from cortex_mem.contract import Caller, Demote, Engine, Event
from cortex_mem.engines.tiers import RuleJudge, TierEngine, knowledge_only
from cortex_mem.runtime import Runtime
from cortex_mem.store.memory import InMemoryEventStore, InMemoryKnowledgeBase, InMemoryStateBackend

T0 = datetime(2026, 11, 1, tzinfo=timezone.utc)
DAY = timedelta(days=1)
FACT = "배포는 main 브랜치 태그로만 한다"
OTHER = "배포는 release 브랜치에서 수동으로 한다"
QUERY = "배포는 어떻게 하나"


def ev(id, text="", kind="record", user="u1", session=None, scope="team", at=T0, signal="self", **kw):
    return Event(id=id, kind=kind, at=at, user=user, team="t1", session=session or f"s-{user}",
                 scope=scope, text=text, signal=signal, **kw)


def verdict(id, refs, outcome="success", signal="test", user="u5", kind="outcome", **kw):
    return ev(id, kw.pop("text", ""), kind=kind, user=user, signal=signal, outcome=outcome, refs=tuple(refs), **kw)


def who(user="u9", at=T0, **kw):
    return Caller(user=user, team="t1", session=f"s-{user}", at=at, **kw)


def runtime(engine=None, **config):
    return Runtime(engine or TierEngine(config), InMemoryEventStore(), InMemoryKnowledgeBase(),
                   InMemoryStateBackend())


def ids(rt):
    return [k.id for k in rt.knowledge.all()]


def demoted(rt):
    return [v.decision for v in rt.verdicts if v.accepted and isinstance(v.decision, Demote)]


def handed(rt, query=QUERY, **caller):
    return [(h.id, h.status) for h in rt.handoff(query, who(**caller), "team")]


def test_is_an_engine():
    assert isinstance(TierEngine(), Engine)
    assert isinstance(knowledge_only(), Engine)


# ── promotion needs an outside check ───────────────────────────────────────


def test_a_self_record_is_a_note_not_knowledge():
    rt = runtime()
    rt.append(ev("r1", FACT))

    assert ids(rt) == []
    [h] = rt.handoff(QUERY, who(), "team")
    assert (h.id, h.status, h.sources) == ("c-r1", "memory", ("r1",))
    assert "미확인 기록" in h.content and "외부 확인 없음" in h.content
    assert h.confidence is None


def test_knowledge_mode_hands_off_no_notes():
    rt = runtime(handoff="knowledge")
    rt.append(ev("r1", FACT))

    assert rt.handoff(QUERY, who(), "team") == []


def test_a_passing_test_makes_it_knowledge():
    rt = runtime()
    rt.append(ev("r1", FACT))
    rt.append(verdict("o1", ["c-r1"], repo="api", rev="abc"))

    assert ids(rt) == ["c-r1"]
    [h] = rt.handoff(QUERY, who(), "team")
    assert h.status == "knowledge" and set(h.sources) == {"r1", "o1"}
    assert "확인된 지식" in h.content and "테스트 1" in h.content


def test_agents_agreeing_with_each_other_never_promote():
    rt = runtime()
    for n, user in enumerate(["u1", "u2", "u3", "u4"]):
        rt.append(ev(f"r{n}", FACT, user=user))
    for n, user in enumerate(["u5", "u6", "u7"]):
        rt.append(verdict(f"o{n}", ["c-r0"], signal="self", user=user))

    assert ids(rt) == []
    [h] = rt.handoff(QUERY, who(), "team")
    assert h.status == "memory" and "7명" in h.content


def test_a_review_or_a_person_makes_it_knowledge():
    for signal, kind in (("review", "human_confirm"), ("human", "human_confirm"), ("doc", "outcome")):
        rt = runtime()
        rt.append(ev("r1", FACT))
        rt.append(verdict("o1", ["r1"], signal=signal, kind=kind, user="u2"))
        assert ids(rt) == ["c-r1"], signal


def test_nobody_approves_their_own_claim():
    rt = runtime()
    rt.append(ev("r1", FACT, user="u1"))
    rt.append(verdict("o1", ["r1"], signal="human", kind="human_confirm", user="u1"))
    rt.append(verdict("o2", ["r1"], signal="review", kind="human_confirm", user="u1"))
    rt.append(verdict("o3", ["r1"], signal="self", user="u1"))

    assert ids(rt) == []
    [h] = rt.handoff(QUERY, who(), "team")
    assert "1명" in h.content


def test_an_authors_test_run_still_counts():
    rt = runtime()
    rt.append(ev("r1", FACT, user="u1"))
    rt.append(verdict("o1", ["r1"], signal="test", user="u1", repo="api", rev="abc"))

    assert ids(rt) == ["c-r1"]


def test_a_record_signed_by_a_test_a_document_or_a_person_is_checked():
    for signal in ("test", "doc", "human"):
        rt = runtime()
        rt.append(ev("r1", FACT, signal=signal))
        assert ids(rt) == ["c-r1"], signal


def test_a_verdicts_text_is_never_a_claim():
    # Contract 3-2: a verdict judges its refs. Whatever else its text says, it
    # confirms the record it points at and states nothing of its own.
    rt = runtime()
    rt.append(ev("r1", FACT))
    rt.append(verdict("h1", ["r1"], signal="human", kind="human_confirm", user="u2", text=OTHER))

    assert ids(rt) == ["c-r1"]
    assert handed(rt) == [("c-r1", "knowledge")]


def test_an_unknown_outcome_changes_nothing():
    rt = runtime()
    rt.append(ev("r1", FACT))
    rt.append(verdict("o1", ["r1"], outcome="unknown"))

    assert ids(rt) == []


# ── the latest outside verdict decides ─────────────────────────────────────


def test_a_later_rejection_demotes_and_a_later_pass_restores():
    rt = runtime()
    rt.append(ev("r1", FACT))
    rt.append(verdict("o1", ["r1"], repo="api", rev="a1", at=T0 + DAY))
    rt.append(verdict("h1", ["c-r1"], outcome="failure", signal="human", kind="human_confirm",
                      user="u2", at=T0 + 2 * DAY))

    assert ids(rt) == []
    [d] = demoted(rt)
    assert (d.id, d.why, d.sources, d.since) == ("c-r1", "invalid", ("h1",), T0 + 2 * DAY)
    assert handed(rt) == []

    rt.append(verdict("o2", ["r1"], repo="api", rev="a2", at=T0 + 3 * DAY))
    assert ids(rt) == ["c-r1"]
    assert set(rt.knowledge.get(["c-r1"])[0].sources) == {"r1", "o1", "o2"}


def test_at_the_same_instant_the_rejection_wins():
    rt = runtime()
    rt.append(ev("r1", FACT))
    rt.append(verdict("o1", ["r1"], repo="api", rev="a1", at=T0 + DAY))
    rt.append(verdict("o2", ["r1"], outcome="failure", repo="api", rev="a2", at=T0 + DAY))

    assert ids(rt) == []


def test_a_refuted_note_is_not_handed_over_and_reposting_does_not_revive_it():
    rt = runtime()
    rt.append(ev("r1", FACT, user="u1"))
    rt.append(verdict("o1", ["r1"], outcome="failure", repo="api", rev="a1"))
    assert handed(rt) == []

    rt.append(ev("r2", FACT, user="u1", at=T0 + DAY))
    rt.append(ev("r3", FACT, user="u3", at=T0 + DAY))
    assert handed(rt) == [] and ids(rt) == []


def test_one_test_run_reported_twice_is_one_check():
    rt = runtime()
    rt.append(ev("r1", FACT))
    rt.append(verdict("o1", ["r1"], repo="api", rev="abc", user="u2"))
    rt.append(verdict("o2", ["r1"], repo="api", rev="abc", user="u3", at=T0 + DAY))

    [h] = rt.handoff(QUERY, who(), "team")
    assert "테스트 1" in h.content and set(h.sources) == {"r1", "o2"}


def test_new_evidence_is_cited():
    rt = runtime()
    rt.append(ev("r1", FACT))
    rt.append(verdict("o1", ["r1"], repo="api", rev="a1"))
    rt.append(verdict("h1", ["r1"], signal="human", kind="human_confirm", user="u2", at=T0 + DAY))

    assert set(rt.knowledge.get(["c-r1"])[0].sources) == {"r1", "o1", "h1"}


# ── handoff tiers ──────────────────────────────────────────────────────────


def test_checked_then_backed_then_single_then_disputed():
    rt = runtime()
    rt.append(ev("d1", "배포는 금요일 오후에 한다", user="u1"))
    rt.append(verdict("f1", ["d1"], outcome="failure", signal="self", user="u8"))
    rt.append(ev("s1", "배포는 수요일 오전에 한다", user="u2"))
    rt.append(ev("b1", "배포는 화요일 저녁에 한다", user="u3"))
    rt.append(ev("b2", "배포는 화요일 저녁에 한다", user="u4"))
    rt.append(ev("k1", "배포는 월요일 새벽에 한다", user="u5"))
    rt.append(verdict("o1", ["k1"], repo="api", rev="a1"))

    assert handed(rt, "배포는 언제 한다") == [
        ("c-k1", "knowledge"), ("c-b1", "memory"), ("c-s1", "memory"), ("c-d1", "memory"),
    ]


def test_a_self_reported_failure_marks_a_note_but_does_not_demote_knowledge():
    rt = runtime()
    rt.append(ev("r1", FACT))
    rt.append(verdict("o1", ["r1"], repo="api", rev="a1"))
    rt.append(verdict("f1", ["r1"], outcome="failure", signal="self", user="u8", at=T0 + DAY))

    assert ids(rt) == ["c-r1"]
    [h] = rt.handoff(QUERY, who(), "team")
    assert h.status == "knowledge" and "실패 보고 1명" in h.content


def test_one_person_repeating_is_one_witness():
    rt = runtime()
    for n in range(4):
        rt.append(ev(f"r{n}", FACT, user="u1", session=f"s{n}", at=T0 + n * DAY))

    [h] = rt.handoff(QUERY, who(), "team")
    assert h.id == "c-r0" and "1명" in h.content


def test_repeating_a_note_does_not_move_it_ahead():
    # The repeated note is newer; the other note matches the query better. No
    # evidence tells them apart, so the search order stands.
    rt = runtime()
    rt.append(ev("t1", "배포는 main 태그로 어떻게 하나 정리", user="u2"))
    for n in range(4):
        rt.append(ev(f"f{n}", "배포는 release 브랜치에서 한다", user="u1", session=f"s{n}", at=T0 + (n + 1) * DAY))

    assert [h for h, _ in handed(rt, at=T0 + 10 * DAY)] == ["c-t1", "c-f0"]


def test_time_alone_changes_nothing():
    rt = runtime()
    rt.append(ev("r1", FACT))
    rt.append(verdict("o1", ["r1"], repo="api", rev="a1"))
    rt.append(ev("r2", OTHER, user="u2"))
    before = rt.handoff(QUERY, who(), "team")

    rt.tick(T0 + 1000 * DAY)

    assert ids(rt) == ["c-r1"] and demoted(rt) == []
    assert rt.handoff(QUERY, who(at=T0 + 1000 * DAY), "team") == before


def test_off_topic_knowledge_does_not_outrank_an_on_topic_note():
    rt = runtime()
    rt.append(ev("k1", "로그는 json 으로 남긴다 배포 와 무관", user="u1"))
    rt.append(verdict("o1", ["k1"], repo="api", rev="a1"))
    rt.append(ev("n1", "배포는 main 브랜치 태그로만 한다 어떻게 하나", user="u2"))

    assert handed(rt, "배포는 main 브랜치 태그로만 어떻게 하나")[0] == ("c-n1", "memory")


def test_conditions_that_contradict_the_question_are_left_out():
    rt = runtime()
    rt.append(ev("r1", FACT, conditions={"service": "api"}))
    rt.append(ev("r2", FACT, user="u2", conditions={"service": "web"}))
    rt.append(ev("r3", OTHER, user="u3"))

    assert [h for h, _ in handed(rt, context={"service": "web"})] == ["c-r2", "c-r3"]
    assert [h for h, _ in handed(rt)] == ["c-r3", "c-r1", "c-r2"]  # unconditional first


# ── independence ───────────────────────────────────────────────────────────


def test_restating_what_was_handed_is_not_a_witness():
    rt = runtime()
    rt.append(ev("r1", FACT, user="u1"))
    rt.recall(QUERY, who("u2"), "team", "q1")
    rt.append(ev("r2", FACT, user="u2"))

    [h] = rt.handoff(QUERY, who(), "team")
    assert "1명" in h.content


def test_the_same_note_written_independently_is_a_second_witness():
    rt = runtime()
    rt.append(ev("r1", FACT, user="u1"))
    rt.append(ev("r2", FACT, user="u2"))

    [h] = rt.handoff(QUERY, who(), "team")
    assert h.id == "c-r1" and "2명" in h.content


class WordJudge(RuleJudge):
    """Stand-in for a model judge: links notes that share the word 태그."""

    links = True

    def agrees(self, a, b):
        return "태그" in a and "태그" in b


def test_a_linking_judge_makes_authors_witnesses_but_still_does_not_promote():
    rt = runtime(TierEngine(judge=WordJudge()))
    rt.append(ev("r1", FACT, user="u1"))
    rt.append(ev("r2", "릴리스할 때는 태그를 main 에 찍어서 배포", user="u2"))
    rt.append(ev("r3", OTHER, user="u3"))

    assert ids(rt) == []
    got = {h.id: h.content for h in rt.handoff(QUERY, who(), "team")}
    assert "2명" in got["c-r1"] and "1명" in got["c-r3"]
    assert next(iter(got)) == "c-r1"


def test_the_default_judge_links_nothing():
    rt = runtime()
    rt.append(ev("r1", FACT, user="u1"))
    rt.append(ev("r2", "릴리스할 때는 태그를 main 에 찍어서 배포", user="u2"))

    assert all("1명" in h.content for h in rt.handoff("배포 태그", who(), "team"))


# ── corrections ────────────────────────────────────────────────────────────


def test_whoever_was_handed_a_refuted_claim_gets_a_correction():
    rt = runtime()
    rt.append(ev("r1", OTHER, user="u1"))
    rt.append(ev("r2", FACT, user="u2"))
    rt.recall(QUERY, who("u3"), "team", "q1")
    rt.append(verdict("h1", ["r1"], outcome="failure", signal="human", kind="human_confirm", user="u4",
                      text="release 브랜치 배포는 막혔다", at=T0 + DAY))

    got = rt.handoff(QUERY, who("u3", at=T0 + 2 * DAY), "team")
    assert [(h.id, h.status) for h in got] == [("c-r2", "memory"), ("x-c-r1", "invalidated")]
    assert got[1].sources == ("h1",) and "release 브랜치 배포는 막혔다" in got[1].content

    assert handed(rt, user="u7", at=T0 + 2 * DAY) == [("c-r2", "memory")]  # never handed it: nothing owed


# ── scope ──────────────────────────────────────────────────────────────────


def test_a_personal_record_never_becomes_team_knowledge():
    rt = runtime()
    rt.append(ev("r1", FACT, user="u1", scope="personal"))
    rt.append(verdict("o1", ["r1"], user="u1", scope="personal", repo="api", rev="a1"))

    [item] = rt.knowledge.all()
    assert (item.scope, item.owner) == ("personal", "u1")
    assert rt.handoff(QUERY, who("u2"), "team") == []
    assert [h.id for h in rt.handoff(QUERY, who("u1"), "team")] == ["c-r1"]


def test_evidence_the_audience_cannot_see_does_not_count():
    rt = runtime()
    rt.append(ev("r1", FACT, user="u1"))
    rt.append(verdict("o1", ["r1"], user="u2", scope="personal", repo="api", rev="a1"))

    assert ids(rt) == [] and rt.rejections == []


def test_the_same_words_in_another_scope_are_another_claim():
    rt = runtime()
    rt.append(ev("r1", FACT, user="u1", scope="personal"))
    rt.append(ev("r2", FACT, user="u1"))

    assert [h for h, _ in handed(rt, user="u1")] == ["c-r1", "c-r2"]
    assert [h for h, _ in handed(rt, user="u2")] == ["c-r2"]


# ── determinism ────────────────────────────────────────────────────────────


def test_the_same_events_give_the_same_handoff():
    def play():
        rt = runtime()
        rt.append(ev("r1", FACT, user="u1"))
        rt.append(ev("r2", OTHER, user="u2"))
        rt.append(ev("r3", FACT, user="u3", at=T0 + DAY))
        rt.append(verdict("o1", ["r2"], outcome="failure", signal="self", user="u4", at=T0 + DAY))
        rt.append(verdict("o2", ["r1"], repo="api", rev="a1", at=T0 + 2 * DAY))
        rt.tick(T0 + 3 * DAY)
        return rt.handoff(QUERY, who(at=T0 + 3 * DAY), "team"), ids(rt)

    assert play() == play()


# ── an agent says its note replaces what it was handed ─────────────────────


def replaced(rt, check_new=True, record_first=True):
    """u1's FACT is knowledge; u2 is handed it, reports it failing and writes OTHER citing it."""
    rt.append(ev("r1", FACT))
    rt.append(verdict("o1", ["c-r1"], at=T0 + DAY))
    steps = [ev("r2", OTHER, user="u2", refs=("c-r1",), at=T0 + 2 * DAY),
             verdict("f1", ["c-r1"], outcome="failure", signal="self", user="u2", at=T0 + 2 * DAY)]
    for step in steps if record_first else reversed(steps):
        rt.append(step)
    if check_new:
        rt.append(verdict("o2", ["c-r2"], at=T0 + 3 * DAY))


def test_saying_it_replaces_knowledge_changes_nothing_by_itself():
    rt = runtime()
    replaced(rt, check_new=False)

    assert ids(rt) == ["c-r1"] and demoted(rt) == []
    assert handed(rt) == [("c-r1", "knowledge"), ("c-r2", "memory")]


def test_once_the_replacement_is_checked_the_old_knowledge_falls():
    for record_first in (True, False):
        rt = runtime()
        replaced(rt, record_first=record_first)

        assert ids(rt) == ["c-r2"]
        [d] = demoted(rt)
        assert (d.id, d.why, d.sources) == ("c-r1", "invalid", ("o2",))
        assert handed(rt) == [("c-r2", "knowledge")]


def test_a_replacement_declared_after_its_check_takes_effect_when_declared():
    rt = runtime()
    rt.append(ev("r1", FACT))
    rt.append(verdict("o1", ["c-r1"], at=T0 + DAY))
    rt.append(ev("r2", OTHER, user="u2", refs=("c-r1",), at=T0 + 2 * DAY))
    rt.append(verdict("o2", ["c-r2"], at=T0 + 3 * DAY))
    assert ids(rt) == ["c-r1", "c-r2"]  # cited, not denied: both stand

    rt.append(verdict("f1", ["c-r1"], outcome="failure", signal="self", user="u2", at=T0 + 4 * DAY))

    assert ids(rt) == ["c-r2"]


def test_two_checked_claims_nobody_tied_together_both_stand():
    rt = runtime()
    rt.append(ev("r1", FACT))
    rt.append(verdict("o1", ["c-r1"], at=T0 + DAY))
    rt.append(ev("r2", OTHER, user="u2", at=T0 + 2 * DAY))
    rt.append(verdict("f1", ["c-r1"], outcome="failure", signal="self", user="u2", at=T0 + 2 * DAY))
    rt.append(verdict("o2", ["c-r2"], at=T0 + 3 * DAY))

    assert ids(rt) == ["c-r1", "c-r2"]


def test_someone_elses_failure_report_does_not_make_my_note_a_replacement():
    rt = runtime()
    rt.append(ev("r1", FACT))
    rt.append(verdict("o1", ["c-r1"], at=T0 + DAY))
    rt.append(ev("r2", OTHER, user="u2", refs=("c-r1",), at=T0 + 2 * DAY))
    rt.append(verdict("f1", ["c-r1"], outcome="failure", signal="self", user="u3", at=T0 + 2 * DAY))
    rt.append(verdict("o2", ["c-r2"], at=T0 + 3 * DAY))

    assert ids(rt) == ["c-r1", "c-r2"]


def test_if_the_replacement_falls_the_old_knowledge_comes_back():
    rt = runtime()
    replaced(rt)
    rt.append(verdict("o3", ["c-r2"], outcome="failure", signal="review", user="u6", at=T0 + 4 * DAY))

    assert ids(rt) == ["c-r1"]
    assert handed(rt) == [("c-r1", "knowledge")]


def test_a_later_check_of_the_old_knowledge_restores_it():
    rt = runtime()
    replaced(rt)
    rt.append(verdict("o3", ["c-r1"], signal="human", user="u6", kind="human_confirm", at=T0 + 4 * DAY))

    assert ids(rt) == ["c-r1", "c-r2"]


def test_replacing_an_unchecked_note_drops_it_once_the_replacement_is_checked():
    rt = runtime()
    rt.append(ev("r1", FACT))
    rt.append(ev("r2", OTHER, user="u2", refs=("c-r1",), at=T0 + DAY))
    rt.append(verdict("f1", ["c-r1"], outcome="failure", signal="self", user="u2", at=T0 + DAY))
    assert handed(rt) == [("c-r2", "memory"), ("c-r1", "memory")]  # disputed goes last

    rt.append(verdict("o2", ["c-r2"], at=T0 + 2 * DAY))

    assert handed(rt) == [("c-r2", "knowledge")]


def test_the_correction_names_what_replaced_it():
    rt = runtime()
    rt.append(ev("r1", FACT))
    rt.append(verdict("o1", ["c-r1"], at=T0 + DAY))
    rt.append(ev("q1", QUERY, kind="recall", user="u9", scope="personal", refs=("c-r1",), at=T0 + DAY))
    rt.append(ev("r2", OTHER, user="u2", refs=("c-r1",), at=T0 + 2 * DAY))
    rt.append(verdict("f1", ["c-r1"], outcome="failure", signal="self", user="u2", at=T0 + 2 * DAY))
    rt.append(verdict("o2", ["c-r2"], at=T0 + 3 * DAY))

    first, second = rt.handoff(QUERY, who("u9"), "team")
    assert (first.id, second.id, second.status) == ("c-r2", "x-c-r1", "invalidated")
    assert "대체" in second.content and OTHER in second.content


def test_claims_that_replace_each_other_do_not_loop():
    rt = runtime()
    rt.append(ev("r1", FACT))
    rt.append(ev("r2", OTHER, user="u2", refs=("c-r1",), at=T0 + DAY))
    rt.append(verdict("f1", ["c-r1"], outcome="failure", signal="self", user="u2", at=T0 + DAY))
    rt.append(ev("r3", FACT, user="u1", session="s-again", refs=("c-r2",), at=T0 + 2 * DAY))
    rt.append(verdict("f2", ["c-r2"], outcome="failure", signal="self", user="u1", session="s-again", at=T0 + 2 * DAY))
    rt.append(verdict("o1", ["c-r1"], at=T0 + 3 * DAY))
    rt.append(verdict("o2", ["c-r2"], at=T0 + 4 * DAY))

    assert ids(rt) == ["c-r2"]  # the later check decides
