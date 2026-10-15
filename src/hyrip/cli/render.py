"""Rich rendering: everything a CLI screen prints, in one module."""

from __future__ import annotations

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..models.article import Article
from ..models.grading import SessionReport
from ..models.questions import GenerationMeta
from ..store.stats import Stats

_VERDICT_BADGES = {
    "correct": "[green]✓[/green]",
    "partial": "[yellow]≈[/yellow]",
    "incorrect": "[red]✗[/red]",
    "skipped": "[dim]·[/dim]",
}


def print_banner(console: Console, article: Article, *, model: str, mode: str, n: int) -> None:
    console.print(
        Panel.fit(
            f"[bold]{article.title}[/bold]\n"
            f"article: {article.word_count} words · "
            f"source: {article.source.value}"
            + (f" · {article.source_url}" if article.source_url else ""),
            title="hyrip — have you read it properly?",
            border_style="cyan",
        )
    )
    console.print(
        f"model [cyan]{model}[/cyan] · mode {mode} · {n} questions · "
        "writing questions now (30-120 s)…"
    )


def print_generation_meta(
    console: Console, meta: GenerationMeta | None, warnings: list[str]
) -> None:
    if meta is None:
        return
    bits = [f"{meta.latency_s:.1f}s"]
    if meta.repair_used:
        bits.append("repair round-trip used")
    if meta.finish_reason == "length":
        bits.append("regenerated at a larger token budget")
    if meta.evidence_unverified:
        bits.append(f"[yellow]{meta.evidence_unverified} evidence quote(s) not verbatim[/yellow]")
    console.print("[green]✓[/green] paper ready — " + ", ".join(bits))
    for warning in warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")


def article_text(article: Article) -> str:
    lines = [
        f"{label}. {paragraph}"
        for label, paragraph in zip(article.paragraph_labels, article.paragraphs, strict=True)
    ]
    return f"{article.title}\n\n" + "\n\n".join(lines)


def print_group_header(
    console: Console, group_no: int, instructions: str, headings: list | None
) -> None:
    console.print(f"[bold]Questions {group_no}[/bold] — {instructions}")
    if headings:
        console.print("[dim]" + "\n".join(headings) + "[/dim]")


def print_question(console: Console, question: dict) -> None:
    console.print()
    console.print(f"[bold]Q{question['number']}.[/bold] {question['prompt']}")
    match question:
        case {"question_type": "multiple_choice" | "summary_choice"}:
            for letter, option in zip("ABCD", question["options"], strict=False):
                console.print(f"  {letter}) {option}")
        case {"question_type": "matching_headings", "paragraph_label": label}:
            console.print(f"  paragraph {label} — answer with a roman numeral")
        case {"question_type": "true_false_not_given"}:
            console.print("  [dim]TRUE / FALSE / NOT GIVEN[/dim]")
        case {"question_type": "yes_no_not_given"}:
            console.print("  [dim]YES / NO / NOT GIVEN[/dim]")
        case {"question_type": "match_paragraphs"}:
            console.print("  [dim]answer with a paragraph letter, e.g. D[/dim]")
        case _:
            if "word_limit" in question:
                console.print(f"  [dim](no more than {question['word_limit']} words)[/dim]")


def footer_text(elapsed_s: int, budget_min: int, n_now: int, n_total: int, skipped: int) -> str:
    mark = "amber" if elapsed_s > budget_min * 60 else "dim"
    return (
        f"[{mark}]Q {n_now}/{n_total}[/] · "
        f"[{mark}]{elapsed_s // 60:02d}:{elapsed_s % 60:02d} elapsed"
        f" / {budget_min:02d}:00 budget[/] · {skipped} skipped"
    )


def print_report(console: Console, report: SessionReport, *, mode: str) -> None:
    score = report.score
    minutes, seconds = divmod(report.elapsed_seconds, 60)
    console.print(
        Panel.fit(
            f"[bold]{score.raw_correct}/{score.total}[/bold] raw · "
            f"[bold cyan]band ≈ {score.band:.1f}[/bold cyan]\n"
            f"{minutes}m {seconds}s{mode_suffix(mode)}\n"
            f"[dim]{score.disclaimer}[/dim]",
            title="report",
            border_style="green",
        )
    )
    if report.observations:
        console.print("[bold]What your reading did[/bold]")
        for observation in report.observations:
            numbers = ", ".join(f"Q{n}" for n in observation.question_numbers)
            console.print(f"  [bold]{observation.title}[/bold] ({numbers})")
            console.print(f"  {observation.detail}")
    if report.next_focus:
        console.print()
        console.print("[bold]Next focus[/bold]")
        for focus in report.next_focus:
            console.print(f"  • {focus}")

    table = Table(title="Question review")
    table.add_column("#", justify="right")
    table.add_column("type")
    table.add_column("verdict")
    table.add_column("you", overflow="fold", max_width=28)
    table.add_column("key", overflow="fold", max_width=28)
    detail_col = "why / evidence"
    table.add_column(detail_col, overflow="fold", max_width=44, no_wrap=False)
    for review in report.reviews:
        table.add_row(
            str(review.number),
            _SHORT_TYPE.get(review.question_type, review.question_type),
            _VERDICT_BADGES[review.verdict],
            review.user_raw or ("— skipped" if review.verdict == "skipped" else ""),
            review.answer_key,
            _review_detail(review.why, review.evidence, review.issues),
        )
    console.print(table)


_SHORT_TYPE = {
    "multiple_choice": "MC",
    "true_false_not_given": "TFNG",
    "yes_no_not_given": "YNNG",
    "matching_headings": "headings",
    "match_paragraphs": "paragraph",
    "completion": "complete",
    "short_answer": "short",
    "summary_choice": "summary",
}


def _review_detail(why: str, evidence, issues: list[str]) -> str:
    bits: list[str] = []
    for issue in issues:
        bits.append(f"[yellow]{issue}[/yellow]")
    if evidence:
        quote = evidence[0].quote
        mark = "✓" if evidence[0].quote_verified else "≈"
        bits.append(f"{mark} ({evidence[0].paragraph}) “{quote}”")
    if why:
        bits.append(why)
    return "\n".join(bits)


def mode_suffix(mode: str) -> str:
    return "" if mode == "exam" else " · tutor"


def print_models(console: Console, models, default: str) -> None:
    table = Table(title="Ollama Cloud models")
    table.add_column("name")
    for model in models:
        mark = " ⭐ default" if model.name == default else ""
        table.add_row(model.name + mark)
    console.print(table)


def doctor_report(console: Console, client, config, db_file) -> bool:
    """Check key presence, keyless catalog, authenticated /v1, and the database."""
    ok = True

    def line(mark: str, text: str) -> None:
        console.print(f"{mark} {text}")

    if config.api_key:
        line("[green]✓[/green]", "OLLAMA_API_KEY is set")
    else:
        ok = False
        line("[yellow]![/yellow]", "OLLAMA_API_KEY is not set")
        console.print(
            "      [dim]export it from ollama.com/settings/keys; "
            "generation will fail until it is[/dim]"
        )

    try:
        models = client.list_models()
        names = {m.name for m in models}
        if config.model in names:
            line(
                "[green]✓[/green]",
                f"catalog reachable — {len(models)} models, default {config.model} present",
            )
        else:
            line(
                "[yellow]![/yellow]",
                f"catalog reachable, but default model "
                f"{config.model} is not in it ({len(models)} models)",
            )
    except Exception as exc:
        ok = False
        line("[red]✗[/red]", f"catalog unreachable: {exc}")

    if config.api_key:
        try:
            with httpx.Client(timeout=10.0) as probe:
                response = probe.get(
                    f"{config.base_url}/models", headers=client._headers(need_key=True)
                )
                if response.status_code == 401:
                    line("[red]✗[/red]", "API key rejected by ollama.com/v1")
                    ok = False
                else:
                    line(
                        "[green]✓[/green]",
                        f"v1 endpoint reachable with key (HTTP {response.status_code})",
                    )
        except Exception as exc:
            ok = False
            line("[red]✗[/red]", f"v1 endpoint unreachable: {exc}")

    try:
        from ..store.db import Store

        store = Store(db_file)
        line("[green]✓[/green]", f"database ready at {db_file} (schema v{store.schema_version})")
    except Exception as exc:
        ok = False
        line("[red]✗[/red]", f"database problem: {exc}")
    return ok


def print_history(console: Console, rows: list[dict]) -> None:
    if not rows:
        console.print("No sessions yet — try [bold]hyrip read[/bold].")
        return
    table = Table(title="Session history")
    table.add_column("id", justify="right")
    table.add_column("mode")
    table.add_column("status")
    table.add_column("passage", overflow="fold", max_width=40)
    table.add_column("score", justify="center")
    table.add_column("band", justify="center")
    table.add_column("elapsed")
    for row in rows:
        correct, total = row["score_correct"], row["score_total"]
        score = f"{correct}/{total}" if correct is not None else "—"
        band = f"{row['band_estimate']:.1f}" if row["band_estimate"] is not None else "—"
        if row["elapsed_seconds"] is not None:
            minutes, seconds = divmod(row["elapsed_seconds"], 60)
            elapsed = f"{minutes:02d}:{seconds:02d}"
        else:
            elapsed = "—"
        table.add_row(
            str(row["id"]),
            row["mode"],
            row["status"],
            row["passage_title"] or "—",
            score,
            band,
            elapsed,
        )
    console.print(table)


def print_transcript(console: Console, transcript: list[dict]) -> None:
    for message in transcript:
        role = "you" if message["role"] == "user" else "tutor"
        style = "dim" if role == "you" else "cyan"
        label = f"[{style}]{role}[/]"
        qnum = message.get("question_number")
        question = f" (Q{qnum})" if qnum else ""
        # content is user-written text; markup=False so "[answer skipped]"-style
        # prefixes render literally instead of being swallowed as rich tags
        console.print(f"{label}{question}: ", end="")
        console.print(message["content"], markup=False)


def print_stats(console: Console, stats: Stats) -> None:
    if not stats.sessions and not stats.per_type:
        console.print("No data yet — finish a session first.")
        return
    if stats.per_type:
        table = Table(title="Accuracy by question type")
        table.add_column("type")
        table.add_column("correct/attempted", justify="center")
        table.add_column("%", justify="right")
        for entry in stats.per_type:
            name = _SHORT_TYPE.get(entry.question_type, entry.question_type)
            table.add_row(
                name,
                f"{entry.correct}/{entry.attempted}",
                f"{entry.pct:.0f}%",
            )
        console.print(table)
    console.print(
        f"sessions: {stats.total_sessions} · answers: {stats.total_questions_answered} · "
        f"correct: {stats.total_correct}"
    )
    recent = Table(title="Recent sessions")
    recent.add_column("id", justify="right")
    recent.add_column("mode")
    recent.add_column("score", justify="center")
    recent.add_column("band", justify="center")
    recent.add_column("passage", overflow="fold", max_width=36)
    for row in stats.sessions:
        score = f"{row.score_correct}/{row.score_total}" if row.score_correct is not None else "—"
        band = f"{row.band_estimate:.1f}" if row.band_estimate is not None else "—"
        recent.add_row(str(row.session_id), row.mode, score, band, row.passage_title or "—")
    console.print(recent)
