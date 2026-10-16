"""generate_paper: happy path, truncation regen, repair round-trip, double failure."""

from __future__ import annotations

import json

import pytest

from hyrip.engine.generator import generate_paper, mix_recipe, parse_json_response
from hyrip.errors import GenerationError
from hyrip.llm.client import ChatResult
from hyrip.models.llm_contracts import RawGeneratedPaper


def _paper_json(raw_paper) -> str:
    return raw_paper.model_dump_json()


def test_happy_path_builds_paper(fake_llm, raw_paper, article, config):
    fake_llm.responses = [
        ChatResult(content=_paper_json(raw_paper), finish_reason="stop", latency_s=0.5)
    ]
    outcome = generate_paper(fake_llm, article, config=config, n_questions=12)
    assert [q.number for q in outcome.paper.questions] == list(range(1, 13))
    assert outcome.paper.main_idea_number == 12
    assert outcome.paper.meta.model == "test-model"
    assert outcome.paper.meta.latency_s == 0.5
    assert outcome.paper.groups[0].group_no == 1
    assert fake_llm.calls[0]["json_mode"] is True
    assert outcome.warnings == []


def test_recipe_sent_in_user_message(fake_llm, raw_paper, article, config):
    fake_llm.responses = [ChatResult(content=_paper_json(raw_paper), finish_reason="stop")]
    generate_paper(fake_llm, article, config=config)
    user = fake_llm.last_user_message
    assert "multiple choice" in user
    assert "12" in user or "2" in user  # the recipe mix is spelled out


def test_truncated_generation_regenerates_with_headroom(fake_llm, raw_paper, article, config):
    fake_llm.responses = [
        ChatResult(content=_paper_json(raw_paper), finish_reason="length", latency_s=0.1),
        ChatResult(content=_paper_json(raw_paper), finish_reason="stop", latency_s=0.2),
    ]
    outcome = generate_paper(fake_llm, article, config=config)
    assert len(fake_llm.calls) == 2
    assert fake_llm.calls[1]["max_tokens"] == int(config.max_tokens * 2)
    assert outcome.paper.meta.finish_reason == "stop"


def test_hard_drift_triggers_repair(fake_llm, raw_paper, article, config):
    """A numbering gap goes back to the model with the error report; repair succeeds."""
    data = json.loads(_paper_json(raw_paper))
    data["groups"][0]["items"][1]["number"] = 3  # numbering jumps 1 → 3
    fake_llm.responses = [
        ChatResult(content=json.dumps(data), finish_reason="stop", latency_s=0.1),
        ChatResult(content=_paper_json(raw_paper), finish_reason="stop", latency_s=0.1),
    ]
    outcome = generate_paper(fake_llm, article, config=config)
    assert len(fake_llm.calls) == 2
    assert outcome.paper.meta.repair_used is True
    roles = [m["role"] for m in fake_llm.calls[1]["messages"]]
    assert roles.count("assistant") == 1
    assert any(
        "failed validation" in m["content"]
        for m in fake_llm.calls[1]["messages"]
        if m["role"] == "user"
    )
    assert outcome.warnings == []


def test_double_failure_raises_GenerationError(fake_llm, raw_paper, article, config):
    data = json.loads(_paper_json(raw_paper))
    data["groups"][0]["items"][1]["number"] = 3
    fake_llm.responses = [
        ChatResult(content=json.dumps(data), finish_reason="stop", latency_s=0.1),
        ChatResult(content=json.dumps(data), finish_reason="stop", latency_s=0.1),
    ]
    with pytest.raises(GenerationError, match="failed validation twice"):
        generate_paper(fake_llm, article, config=config)


def test_schema_drift_repair_gets_real_validator_errors(fake_llm, raw_paper, article, config):
    """extra="forbid" violations must reach the repair call as actionable schema detail,
    not the blanket 'not valid JSON' phrase (the model can't fix what it can't see)."""
    data = json.loads(_paper_json(raw_paper))
    data["groups"][0]["items"][0]["topical"] = "an extra field the contract forbids"
    fake_llm.responses = [
        ChatResult(content=json.dumps(data), finish_reason="stop", latency_s=0.1),
        ChatResult(content=_paper_json(raw_paper), finish_reason="stop", latency_s=0.1),
    ]
    outcome = generate_paper(fake_llm, article, config=config)
    assert outcome.paper.meta.repair_used is True
    repair_user = next(m for m in reversed(fake_llm.calls[1]["messages"]) if m["role"] == "user")
    assert "schema" in repair_user["content"]
    assert "topical" in repair_user["content"]
    assert "not valid paper JSON" not in repair_user["content"]


def test_unparseable_output_regenerates_with_headroom_then_fails(fake_llm, article, config):
    """Unparseable output cannot be repaired by feedback (the model must rewrite from
    scratch anyway) — it regenerates with token headroom instead."""
    fake_llm.responses = [
        ChatResult(content="not json at all", finish_reason="stop"),
        ChatResult(content="still not json", finish_reason="stop"),
    ]
    with pytest.raises(GenerationError):
        generate_paper(fake_llm, article, config=config)
    assert len(fake_llm.calls) == 2
    assert fake_llm.calls[1]["max_tokens"] == int(config.max_tokens * 2)
    roles = [m["role"] for m in fake_llm.calls[1]["messages"]]
    assert roles.count("assistant") == 0  # a fresh generation, not a repair transcript


def test_truncated_json_regenerates_not_repairs(fake_llm, raw_paper, article, config):
    """Output cut mid-object (pydantic: 'Invalid JSON: EOF while parsing a list …') is a
    token-budget problem, not drift: rewrite from scratch with headroom, no repair."""
    text = _paper_json(raw_paper)
    cut = text[: text.rindex('"summary_choice"')]  # ends inside the trailing summary group
    fake_llm.responses = [
        ChatResult(content=cut, finish_reason="stop"),
        ChatResult(content=_paper_json(raw_paper), finish_reason="stop"),
    ]
    outcome = generate_paper(fake_llm, article, config=config)
    assert len(fake_llm.calls) == 2
    assert fake_llm.calls[1]["max_tokens"] == int(config.max_tokens * 2)
    roles = [m["role"] for m in fake_llm.calls[1]["messages"]]
    assert roles.count("assistant") == 0
    assert outcome.paper.meta.repair_used is False
    assert outcome.warnings == []


def test_double_truncation_reports_budget_hint(fake_llm, article, config):
    cut = '{"groups": [{"question_type": "multiple_choice", "items": ['
    fake_llm.responses = [
        ChatResult(content=cut, finish_reason="stop"),
        ChatResult(content=cut, finish_reason="stop"),
    ]
    with pytest.raises(GenerationError, match="HYRIP_MAX_TOKENS"):
        generate_paper(fake_llm, article, config=config)


def test_seven_group_paper_parses(raw_paper):
    """The n=14/16 recipes hold 7 groups (short_answer joins at 14) — the schema must accept
    them. Regression: the contract capped groups at 6, making 14/16 unvalidatable."""
    data = json.loads(_paper_json(raw_paper))
    completion = data["groups"][4]
    data["groups"].append(
        {
            **completion,
            "question_type": "short_answer",
            "items": [completion["items"][0]],
        }
    )
    assert len(json.loads(_paper_json(RawGeneratedPaper.model_validate(data)))["groups"]) == 7


def test_recipe_group_counts_stay_within_the_contract():
    meta = RawGeneratedPaper.model_fields["groups"].metadata
    cap = next(m.max_length for m in meta if getattr(m, "max_length", None) is not None)
    for n in range(8, 17):
        assert len(mix_recipe(n=n)) <= cap


def test_soft_drift_reported_as_warning(fake_llm, raw_paper, article, config):
    """A non-verbatim evidence quote survives to the paper but is reported."""
    data = json.loads(_paper_json(raw_paper))
    data["groups"][0]["items"][0]["evidence"][0]["quote"] = "something not in the passage at all"
    fake_llm.responses = [ChatResult(content=json.dumps(data), finish_reason="stop")]
    outcome = generate_paper(fake_llm, article, config=config)
    assert any("evidence" in w.lower() for w in outcome.warnings)
    assert outcome.paper.meta.evidence_unverified >= 1


def test_parse_json_response_variants():
    fenced = '```json\n{"a": 1}\n```'
    assert parse_json_response(fenced) == '{"a": 1}'
    prose = 'Here you go:\n{"a": 1} — hope it helps'
    assert parse_json_response(prose) == '{"a": 1}'
    with pytest.raises(GenerationError):
        parse_json_response("no braces here")


def test_mix_recipe_defaults():
    recipe = mix_recipe(n=12)
    kinds = dict(recipe)
    assert kinds["summary_choice"] == 1
    assert recipe[-1][0] == "summary_choice"
    assert sum(count for _, count in recipe) == 12


def test_mix_recipe_claims_swaps_tfng():
    recipe = mix_recipe(n=12, claims_mode=True)
    assert dict(recipe).get("yes_no_not_given") == 2
    assert "true_false_not_given" not in dict(recipe)


def test_mix_recipe_small_passage_collapses_matching():
    recipe = mix_recipe(n=10, n_paragraphs=3)
    kinds = dict(recipe)
    assert "matching_headings" not in kinds
    assert "match_paragraphs" not in kinds
    assert kinds["completion"] == 5  # collapsed headings + which-paragraph slots


def test_mix_recipe_bounds():
    with pytest.raises(GenerationError):
        mix_recipe(n=6)
    with pytest.raises(GenerationError):
        mix_recipe(n=18)
    odd = mix_recipe(n=13)  # odd n bumps from the 12 profile
    assert sum(count for _, count in odd) == 13
