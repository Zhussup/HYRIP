"""Band boundaries, report assembly, each deterministic observation rule, coach merge."""

from __future__ import annotations

import json

import pytest

from hyrip.engine.grader import grade_closed, grade_one_open, grade_paper
from hyrip.engine.insights import build_observations, coach_observations, next_focus
from hyrip.engine.report import band_for_pct, build_report, tally_of
from hyrip.llm.client import ChatResult
from hyrip.models.grading import UserAnswer
from hyrip.models.llm_contracts import RawGeneratedPaper
from hyrip.models.questions import build_paper


@pytest.fixture
def paper(raw_paper, article):
    return build_paper(raw_paper, article=article)


def answers(**by_number: str | None) -> dict[int, UserAnswer]:
    return {
        int(n): UserAnswer(number=int(n), raw=v, skipped=v is None or v == "")
        for n, v in by_number.items()
    }


# ---- bands ----------------------------------------------------------------------


def test_band_boundaries():
    assert band_for_pct(100.0) == 9.0
    assert band_for_pct(97.5) == 9.0
    assert band_for_pct(97.4) == 8.0
    assert band_for_pct(87.5) == 8.0
    assert band_for_pct(87.4) == 7.5
    assert band_for_pct(82.5) == 7.5
    assert band_for_pct(75.0) == 7.0
    assert band_for_pct(74.9) == 6.5
    assert band_for_pct(67.5) == 6.5
    assert band_for_pct(57.5) == 6.0
    assert band_for_pct(47.5) == 5.5
    assert band_for_pct(37.5) == 5.0
    assert band_for_pct(32.5) == 4.5
    assert band_for_pct(25.0) == 4.0
    assert band_for_pct(24.9) == 3.5
    assert band_for_pct(0.0) == 3.5


def test_tally(paper):
    graded = {
        1: grade_closed(paper.questions[0], "A", skipped=False),
        2: grade_closed(paper.questions[1], "A", skipped=False),
        10: grade_one_open(paper.questions[9], "30", skipped=False, mode="exam"),
        11: grade_one_open(
            paper.questions[10], "starve the pollinators", skipped=False, mode="tutor"
        ),
        12: grade_closed(paper.questions[11], "B", skipped=False),
    }
    for number in (3, 4, 5, 6, 7, 8, 9):
        graded[number] = grade_closed(paper.questions[number - 1], None, skipped=True)
    tally = tally_of(graded, total=12)
    assert tally.correct == 4
    assert tally.answered == 5
    assert tally.skipped == 7
    assert tally.total == 12


# ---- build_report ---------------------------------------------------------------


def test_build_report_fields(paper):
    graded = grade_paper(
        paper,
        answers(
            **{
                "1": "A",
                "2": "C",
                "3": "TRUE",
                "4": "FALSE",
                "5": "ii",
                "6": "iii",
                "7": "iv",
                "8": "H",
                "9": "D",
                "10": "30",
                "11": "starve the pollinators",
                "12": "B",
            }
        ),
        mode="exam",
    )
    report = build_report(paper, graded, elapsed_seconds=421)
    assert report.score.raw_correct == 12
    assert report.score.total == 12
    assert report.score.pct == 100.0
    assert report.score.band == 9.0
    assert "Unofficial estimate" in report.score.disclaimer
    assert report.elapsed_seconds == 421
    assert len(report.reviews) == 12
    first = report.reviews[0]
    assert first.prompt.startswith("According to")
    assert first.answer_key == "A"
    assert first.verdict == "correct"


def test_build_report_partial_score(paper):
    graded = grade_paper(
        paper,
        answers(
            **{
                "1": "A",
                "2": "B",
                "3": "T",
                "4": "T",
                "5": "ii",
                "6": "iii",
                "7": "1",
                "8": "H",
                "9": "D",
                "10": "30",
                "11": "starve the pollinators",
                "12": "B",
            }
        ),
        mode="exam",
    )
    report = build_report(paper, graded, elapsed_seconds=100)
    # correct: 1, 3(T), 5, 6, 8, 9, 10, 11 (acceptable variant), 12 = 9; 9/12 = 75.0%
    assert report.score.raw_correct == 9
    assert report.score.pct == pytest.approx(75.0)
    assert report.score.band == 7.0
    wrong_reviews = [r for r in report.reviews if r.verdict == "incorrect"]
    assert len(wrong_reviews) == 3
    assert all(r.answer_key for r in report.reviews)


# ---- observation rules -----------------------------------------------------------


@pytest.fixture
def mini_ng_paper(article):
    """Minimal paper with two NOT-GIVEN items to trigger the confusion rule."""
    data = {
        "passage_title": "Mini",
        "groups": [
            {
                "question_type": "true_false_not_given",
                "instructions": "AGREE?",
                "items": [
                    {
                        "number": 1,
                        "prompt": "Statement one.",
                        "answer_key": "NOT GIVEN",
                        "evidence": [],
                        "explanation": "silence",
                    },
                    {
                        "number": 2,
                        "prompt": "Statement two.",
                        "answer_key": "FALSE",
                        "evidence": [{"paragraph": "A", "quote": "the"}],
                        "explanation": "no",
                    },
                    {
                        "number": 3,
                        "prompt": "Statement three.",
                        "answer_key": "NOT GIVEN",
                        "evidence": [],
                        "explanation": "silence",
                    },
                ],
            },
            {
                "question_type": "summary_choice",
                "instructions": "BEST SUMMARY",
                "items": [
                    {
                        "number": 4,
                        "prompt": "Best summary?",
                        "answer_key": "B",
                        "options": [
                            {"text": "One narrow detail.", "trap_kind": "narrow"},
                            {"text": "The broad idea.", "trap_kind": None},
                            {"text": "A wild extrapolation.", "trap_kind": "broad"},
                        ],
                        "evidence": [],
                        "explanation": "B",
                    }
                ],
            },
        ],
    }
    return build_paper(RawGeneratedPaper.model_validate(data), article=article)


def test_not_given_confusion_rule(mini_ng_paper):
    paper = mini_ng_paper
    graded = {
        1: grade_closed(paper.questions[0], "TRUE", skipped=False),
        2: grade_closed(paper.questions[1], "FALSE", skipped=False),
        3: grade_closed(paper.questions[2], "F", skipped=False),
        4: grade_closed(paper.questions[3], "B", skipped=False),
    }
    obs = build_observations(paper, graded)
    kinds = [o.kind for o in obs]
    assert "not_given_confusion" in kinds
    ng_obs = next(o for o in obs if o.kind == "not_given_confusion")
    assert sorted(ng_obs.question_numbers) == [1, 3]


def test_main_idea_trap_notes(paper):
    graded = {12: grade_closed(paper.questions[11], "3", skipped=False)}  # narrow option
    obs = [o for o in build_observations(paper, graded) if o.kind == "main_idea"]
    assert obs, "main-idea observation must always appear for an answered summary item"
    assert "narrow" in obs[0].title
    assert obs[0].question_numbers == [12]


def test_main_idea_correct_note(paper):
    graded = {12: grade_closed(paper.questions[11], "B", skipped=False)}
    obs = [o for o in build_observations(paper, graded) if o.kind == "main_idea"]
    assert obs and "captured" in obs[0].title


def test_skimming_vs_detail_rule(paper):
    answers_map = answers(
        **{
            "1": "A",
            "2": "C",
            "10": "30",
            "11": "starve the pollinators",  # detail: all correct
            "3": "F",
            "4": "T",
            "5": "i",
            "6": "i",
            "7": "i",
            "8": "B",
            "9": "B",  # scan: all wrong
            "12": "B",
        }
    )
    graded = grade_paper(paper, answers_map, mode="exam")
    obs = [o for o in build_observations(paper, graded) if o.kind == "skimming_vs_detail"]
    assert obs
    assert "Detail better" in obs[0].title


def test_word_limit_observation(paper):
    graded = {
        10: grade_one_open(paper.questions[9], "around 30 kg exactly", skipped=False, mode="exam"),
        11: grade_one_open(
            paper.questions[10], "starve all of the wild pollinators", skipped=False, mode="exam"
        ),
    }
    obs = [o for o in build_observations(paper, graded) if o.kind == "word_limit_slips"]
    assert obs and sorted(obs[0].question_numbers) == [10, 11]


def test_observations_capped_at_four(paper):
    every_wrong = answers(**{str(n): "B" for n in range(1, 13)})
    every_wrong[12] = UserAnswer(number=12, raw="C", skipped=False)
    obs = build_observations(paper, grade_paper(paper, every_wrong, mode="exam"))
    assert len(obs) <= 4


def test_next_focus_maps_and_limits(paper):
    wrong = grade_paper(paper, answers(**{str(n): "B" for n in range(1, 13)}), mode="exam")
    obs = build_observations(paper, wrong)
    focus = next_focus(obs)
    assert len(focus) <= 3
    assert all(isinstance(f, str) and len(f) > 12 for f in focus)
    kinds_present = {o.kind for o in obs}
    assert any(k in kinds_present for k in ("paraphrase_trap", "main_idea", "skimming_vs_detail"))


# ---- coach -----------------------------------------------------------------------


def test_coach_appended_when_client_works(paper, article, fake_llm):
    fake_llm.responses = [
        ChatResult(
            content=json.dumps(
                {
                    "observations": [
                        {
                            "kind": "rereading",
                            "title": "Too many returns",
                            "detail": "You re-read paragraph C three times.",
                            "question_numbers": [2, 10],
                        }
                    ]
                }
            ),
            finish_reason="stop",
        )
    ]
    coach = coach_observations(fake_llm, article, [], [])
    assert coach[0].kind == "rereading"
    assert coach[0].question_numbers == [2, 10]


def test_coach_failure_is_silent(paper, article, fake_llm):
    fake_llm.responses = [ChatResult(content="garbage", finish_reason="stop")]
    assert coach_observations(fake_llm, article, [], []) == []


def test_report_merges_coach_behind_deterministic(paper, article, fake_llm):
    graded = grade_paper(paper, answers(**{"12": "B"}), mode="exam")
    report = build_report(paper, graded, elapsed_seconds=60)
    deterministic_kinds = [o.kind for o in report.observations]
    coach = coach_observations(fake_llm, article, report.reviews, report.observations)
    final_kinds = deterministic_kinds + [o.kind for o in coach]
    assert final_kinds[: len(deterministic_kinds)] == deterministic_kinds
    assert final_kinds[len(deterministic_kinds) :] == [o.kind for o in coach]
