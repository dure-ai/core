from datetime import timedelta

import pytest

from cortex_mem.engines.ledger.config import GateRule, LedgerConfig


def test_defaults():
    cfg = LedgerConfig.from_mapping(None)
    assert cfg == LedgerConfig.from_mapping({})
    assert cfg.similar_same_threshold == 0.0
    assert cfg.same_threshold == 0.8
    assert cfg.copy_threshold == 0.9
    assert cfg.match_pool == 10
    assert cfg.pool == 50
    assert cfg.gates["personal"] == GateRule(2, 1, False, timedelta(0), 0.20)
    assert cfg.gates["team"] == GateRule(2, 1, False, timedelta(0), 0.15)
    assert dict(cfg.weights) == {
        "observation": 1.0,
        "review": 1.5,
        "test": 2.0,
        "doc": 2.0,
        "human": 3.0,
    }
    assert cfg.trust_observation == 0.7
    assert cfg.alpha == 1.0
    assert cfg.beta == 1.0
    assert dict(cfg.status_weights) == {
        "candidate": 0.5,
        "provisional": 0.6,
        "established": 1.0,
        "disputed": 0.4,
    }
    assert cfg.own_candidate_weight == 0.7
    assert cfg.review_after == timedelta(0)
    assert cfg.half_life == timedelta(0)
    assert cfg.high_risk_terms == ()


def test_overrides():
    cfg = LedgerConfig.from_mapping(
        {
            "similar.same_threshold": "0.85",
            "same_threshold": "0.7",
            "gate.team.require_verified": "true",
            "gate.team.min_span_days": "3",
            "weight.human": "5",
            "high_risk_terms": "Delete, payment ,",
        }
    )
    assert cfg.similar_same_threshold == 0.85
    assert cfg.same_threshold == 0.7
    assert cfg.gates["team"].require_verified is True
    assert cfg.gates["team"].min_span == timedelta(days=3)
    assert cfg.weights["human"] == 5.0
    assert cfg.high_risk_terms == ("delete", "payment")
    assert cfg.gates["personal"] == GateRule(2, 1, False, timedelta(0), 0.20)
    assert cfg.weights["test"] == 2.0
    assert cfg.pool == 50


def test_unknown_key_is_refused():
    with pytest.raises(ValueError, match="same_treshold"):
        LedgerConfig.from_mapping({"same_treshold": "0.7"})


def test_bad_value_names_the_key():
    with pytest.raises(ValueError, match="pool"):
        LedgerConfig.from_mapping({"pool": "many"})


@pytest.mark.parametrize(
    "bad",
    [
        {"gate.org.min_sources": "2"},
        {"weight.rumor": "1"},
        {"status_weight.gone": "1"},
        {"gate.team.require_verified": "yes"},
        {"gate.team.max_contra_ratio": "abc"},
        {"similar.same_threshold": "-0.1"},
        {"similar.same_threshold": "1.1"},
        {"same_threshold": "0"},
        {"same_threshold": "1.1"},
        {"same_threshold": "0.95"},  # copy_threshold (0.9) would be below it
        {"copy_threshold": "0.7"},  # below same_threshold (0.8)
        {"copy_threshold": "1.1"},
        {"match_pool": "0"},
        {"pool": "0"},
        {"weight.human": "-1"},
        {"status_weight.candidate": "-0.5"},
        {"own_candidate_weight": "-1"},
        {"trust.observation": "-0.1"},
        {"alpha": "-1"},
        {"beta": "-1"},
        {"alpha": "0", "beta": "0"},
        {"gate.team.min_sources": "-1"},
        {"gate.personal.alt_verified": "-1"},
        {"gate.team.max_contra_ratio": "-0.1"},
        {"gate.team.max_contra_ratio": "1.5"},
        {"gate.team.min_span_days": "-1"},
        {"review_after_days": "-1"},
        {"half_life_days": "-1"},
    ],
)
def test_ranges(bad):
    with pytest.raises(ValueError) as exc:
        LedgerConfig.from_mapping(bad)
    key = next(iter(bad))
    if key != "same_threshold" or bad[key] != "0.95":
        assert key in str(exc.value)


def test_mappings_are_read_only():
    cfg = LedgerConfig.from_mapping(None)
    with pytest.raises(TypeError):
        cfg.weights["human"] = 9
    with pytest.raises(TypeError):
        cfg.status_weights["candidate"] = 9
    with pytest.raises(TypeError):
        cfg.gates["team"] = cfg.gates["personal"]


def test_one_zero_prior_is_allowed():
    assert LedgerConfig.from_mapping({"alpha": "0"}).alpha == 0.0
    assert LedgerConfig.from_mapping({"beta": "0"}).beta == 0.0
