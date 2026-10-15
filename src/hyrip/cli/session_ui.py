"""The interactive CLI question loop. `input_fn` is injectable for tests."""

from __future__ import annotations

from collections.abc import Callable

from rich.console import Console

from ..errors import LlmError
from ..models.grading import UserAnswer
from ..service.session import SessionService
from . import render
from .answers import HELP_TEXT, parse_answer

NEXT_ACTION = Callable[[], bool]  # True when the loop should end


def run_session(
    console: Console,
    service: SessionService,
    session_id: int,
    article,
    *,
    input_fn: Callable[[str], str] = input,
    pager: Callable[[str], None] | None = None,
) -> bool:
    """Drive a session to finished/abandoned. Returns True when a report was produced."""
    show = pager or (lambda text: console.print(text))
    total = service.view(session_id)["n_questions"]
    index = 0
    last_group = 0

    try:
        while index < total:
            view = service.view(session_id)
            question = next(q for q in view["questions"] if q["number"] == index + 1)
            if question["group_no"] != last_group:
                _print_group(console, view, question["group_no"])
                last_group = question["group_no"]

            render.print_question(console, question)
            mode, target = _ask_one(
                console,
                service,
                session_id,
                question,
                article,
                index=index,
                total=total,
                input_fn=input_fn,
                show_article=show,
            )
            match mode:
                case "goto":
                    index = target
                case "advance":
                    index += 1
                case "finish":
                    break
                case "quit":
                    service.abandon(session_id)
                    console.print("[yellow]Session saved as abandoned; see hyrip history.[/yellow]")
                    return False
    except (KeyboardInterrupt, EOFError):
        service.abandon(session_id)
        console.print("\n[yellow]Interrupted — session saved as abandoned.[/yellow]")
        return False

    report = service.finish(session_id)
    render.print_report(console, report, mode=service.snapshot(session_id).mode)
    return True


def _ask_one(
    console: Console,
    service: SessionService,
    session_id: int,
    question: dict,
    article,
    *,
    index: int,
    total: int,
    input_fn: Callable[[str], str],
    show_article: Callable[[str], None],
) -> tuple[str, int | None]:
    from .render import article_text

    mode_str = "tutor" if service.snapshot(session_id).mode == "tutor" else "exam"
    while True:
        snap = service.snapshot(session_id)
        if snap.status != "active":
            console.print()
            console.print("[dim]reading pause over — clock already running[/dim]")
        _print_footer(console, snap, index, total)
        line = input_fn("  answer> ")
        action = parse_answer(line)
        match action.kind:
            case "value":
                if not action.raw:
                    console.print("[dim](blank — answer, or ? for help)[/dim]")
                    continue
                graded = service.record_answer(
                    session_id, UserAnswer(number=question["number"], raw=action.raw)
                )
                if mode_str == "tutor":
                    _tutor_round(console, service, session_id, question["number"], graded, input_fn)
                if question["number"] == total:
                    console.print("[dim]that was the last question — press f to finish[/dim]")
                    continue
                return "advance", None
            case "skip":
                service.record_answer(
                    session_id, UserAnswer(number=question["number"], skipped=True)
                )
                console.print("[dim]skipped[/dim]")
                return "advance", None
            case "back":
                return "goto", max(0, index - 1)
            case "finish":
                if _confirm(console, input_fn):
                    return "finish", None
                continue
            case "view":
                show_article(article_text(article))
                continue
            case "list":
                _print_group(console, service.view(session_id), question["group_no"])
                continue
            case "help":
                console.print(HELP_TEXT)
                continue
            case "quit":
                return "quit", None
    # unreachable: every action returns above


def _tutor_round(console, service, session_id, number, graded, input_fn) -> None:
    verdict_style = {"correct": "green", "partial": "yellow", "incorrect": "red", "skipped": "dim"}[
        graded.verdict
    ]
    console.print(f"  [{verdict_style}]{graded.verdict}[/] (marked by {graded.graded_by})")
    try:
        for chunk in service.tutor_feedback_stream(session_id, number, graded):
            console.print(chunk, end="")
        console.print()
    except LlmError as exc:
        console.print(f"[yellow]tutor feedback unavailable: {exc.user_message()}[/yellow]")
        return
    while True:
        text = input_fn("  tutor? (blank to continue) ")
        if not text.strip():
            return
        try:
            for chunk in service.tutor_followup_stream(session_id, number, text):
                console.print(chunk, end="")
            console.print()
        except LlmError as exc:
            console.print(f"[yellow]tutor unavailable: {exc.user_message()}[/yellow]")
            return


def _confirm(console, input_fn) -> bool:
    answer = input_fn("  finish and mark the paper? [y/N] ")
    return answer.strip().lower() in ("y", "yes")


def _print_footer(console, snap, index, total) -> None:
    console.print(
        render.footer_text(
            snap.elapsed_seconds or 0,
            snap.time_budget_min,
            index + 1,
            total,
            snap.n_skipped,
        )
    )


def _print_group(console, view: dict, group_no: int) -> None:
    group = next(g for g in view["groups"] if g["group_no"] == group_no)
    console.print()
    console.print(f"[bold]Group {group_no}[/bold] — {group['instructions']}")
    if group.get("headings"):
        console.print("[dim]Heading list (l re-prints):[/dim]")
        for heading in group["headings"]:
            console.print(f"  {heading}")
