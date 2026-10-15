"""Command line entry point: read|file|url|models|history|stats|serve|gui|doctor."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from ..config import Config
from ..errors import HyripError
from ..version import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hyrip",
        description="Have you read it properly? IELTS reading trainer over Ollama Cloud.",
    )
    parser.add_argument("--version", action="version", version=f"hyrip {__version__}")
    sub = parser.add_subparsers(dest="command", required=False)

    def session_options(sp) -> None:
        sp.add_argument("--mode", choices=("exam", "tutor"), default="exam")
        sp.add_argument("--model", default=None)
        sp.add_argument("--questions", type=int, default=None, help="8-16")
        sp.add_argument("--time", type=int, default=None, help="time budget in minutes")
        sp.add_argument(
            "--claims", action="store_true", help="YES/NO/NOT GIVEN flavour instead of TF/NG"
        )
        sp.add_argument(
            "--save-raw",
            default=None,
            metavar="PATH",
            help="write the raw generation response for post-mortems",
        )

    read_parser = sub.add_parser("read", help="paste an article, then get IELTS questions")
    session_options(read_parser)

    file_parser = sub.add_parser("file", help="read an article from a local .txt/.md file")
    file_parser.add_argument("path")
    session_options(file_parser)

    url_parser = sub.add_parser("url", help="read an article from a web page")
    url_parser.add_argument("url")
    session_options(url_parser)

    sub.add_parser("models", help="list the Ollama Cloud catalog")
    history = sub.add_parser("history", help="list sessions; replay one")
    history.add_argument("session_id", nargs="?", type=int)
    sub.add_parser("stats", help="accuracy by type + recent sessions")
    serve = sub.add_parser("serve", help="run the web interface")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--host", default="127.0.0.1")
    sub.add_parser("gui", help="run the desktop (Qt) interface")
    sub.add_parser("doctor", help="check key, network and database")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = _console()
    try:
        return _dispatch(args, console)
    except (HyripError, KeyboardInterrupt) as exc:
        _print_error(console, exc)
        return 1


def _console():
    from rich.console import Console

    return Console()


def _print_error(console, exc) -> None:
    message = getattr(exc, "user_message", None)
    console.print(f"[red]error:[/red] {message() if message else str(exc)}")
    hint = getattr(exc, "hint", None)
    if hint:
        console.print(f"[dim]hint: {hint}[/dim]")


def _dispatch(args, console) -> int:
    if args.command is None:
        build_parser().print_help()
        return 0
    handlers: dict[str, Callable[[argparse.Namespace, object], int]] = {
        "doctor": _cmd_doctor,
        "models": _cmd_models,
        "history": _cmd_history,
        "stats": _cmd_stats,
        "serve": _cmd_serve,
        "gui": _cmd_gui,
        "read": _cmd_article,
        "file": _cmd_article,
        "url": _cmd_article,
    }
    return handlers[args.command](args, console)


# ---- article commands ---------------------------------------------------------------


def _cmd_article(args, console) -> int:
    from ..llm.client import OllamaCloudClient
    from ..paths import db_path
    from ..service.session import SessionService
    from ..store.db import Store
    from ..store.repo import load_article, load_paper, load_paper_raw, load_session
    from .render import article_text, print_banner, print_generation_meta
    from .session_ui import run_session

    config = Config.from_env()
    service = SessionService(Store(db_path()), OllamaCloudClient(config), config)

    if args.command == "read":
        console.print(
            "Paste the article body; end with a line saying [bold]EOF[/bold] (or Ctrl-D)."
        )
        lines: list[str] = []
        try:
            for line in sys.stdin:
                if line.rstrip() == "EOF":
                    break
                lines.append(line.rstrip())
        except EOFError:
            pass
        article_id = service.create_article_from_text("\n".join(lines))
    elif args.command == "file":
        article_id = service.create_article_from_file(args.path)
    else:
        article_id = service.create_article_from_url(args.url)

    article = load_article(service.store, article_id)
    session_id = service.start_and_generate(
        article_id,
        mode=args.mode,
        model=args.model,
        n_questions=args.questions,
        claims=args.claims,
        time_budget_min=args.time,
    )

    if args.save_raw:
        paper_id = load_session(service.store, session_id).paper_id
        raw = load_paper_raw(service.store, paper_id) if paper_id is not None else None
        if raw:
            from pathlib import Path

            Path(args.save_raw).write_text(raw, encoding="utf-8")
            console.print(f"[dim]raw response saved to {args.save_raw}[/dim]")

    paper = load_paper(service.store, load_session(service.store, session_id).paper_id)
    print_banner(
        console,
        article,
        model=paper.meta.model if paper.meta else "",
        mode=args.mode,
        n=len(paper.questions),
    )
    print_generation_meta(console, paper.meta, warnings=[])

    def _pager(text: str) -> None:
        with console.pager():
            console.print(text)

    console.print("[dim]Article opens in the pager — q closes it, v reopens it later.[/dim]")
    _pager(article_text(article))
    service.start(session_id)
    run_session(console, service, session_id, article, pager=_pager)
    return 0


def _cmd_models(_, console) -> int:
    from ..llm.client import OllamaCloudClient
    from .render import print_models

    config = Config.from_env()
    print_models(console, OllamaCloudClient(config).list_models(), config.model)
    return 0


def _cmd_history(args, console) -> int:
    from ..errors import StateError
    from ..llm.client import OllamaCloudClient
    from ..paths import db_path
    from ..service.session import SessionService
    from ..store.db import Store
    from .render import print_history, print_report, print_transcript

    service = SessionService(
        Store(db_path()), OllamaCloudClient(Config.from_env()), Config.from_env()
    )
    rows = service.list_history()
    if args.session_id is None:
        print_history(console, rows)
        return 0
    if args.session_id not in [row["id"] for row in rows]:
        console.print(f"[red]error:[/red] session {args.session_id} is not in history")
        return 1
    try:
        report = service.report_of(args.session_id)
    except StateError:
        console.print(f"[red]error:[/red] session {args.session_id} has no report")
        return 1
    snap = service.snapshot(args.session_id)
    print_report(console, report, mode=snap.mode)
    if snap.mode == "tutor":
        print_transcript(console, service.transcript(args.session_id))
    return 0


def _cmd_stats(_, console) -> int:
    from ..llm.client import OllamaCloudClient
    from ..paths import db_path
    from ..service.session import SessionService
    from ..store.db import Store
    from .render import print_stats

    service = SessionService(
        Store(db_path()), OllamaCloudClient(Config.from_env()), Config.from_env()
    )
    print_stats(console, service.stats())
    return 0


def _cmd_serve(args, console) -> int:
    from ..llm.client import OllamaCloudClient
    from ..paths import db_path
    from ..service.session import SessionService
    from ..store.db import Store

    service = SessionService(
        Store(db_path()), OllamaCloudClient(Config.from_env()), Config.from_env()
    )
    try:
        from ..web.app import run_server
    except ModuleNotFoundError:
        console.print(
            "[red]error:[/red] web extras missing — install with `pip install -e '.\\[web]'`"
        )
        return 1
    run_server(service, host=args.host, port=args.port)
    return 0


def _cmd_gui(_, console) -> int:
    from ..llm.client import OllamaCloudClient
    from ..paths import db_path
    from ..service.session import SessionService
    from ..store.db import Store

    service = SessionService(
        Store(db_path()), OllamaCloudClient(Config.from_env()), Config.from_env()
    )
    try:
        from ..qt.app import run_gui
    except ModuleNotFoundError:
        console.print(
            "[red]error:[/red] qt extras missing — install with `pip install -e '.\\[qt]'`"
        )
        return 1
    run_gui(service)
    return 0


def _cmd_doctor(args, console) -> int:
    from ..llm.client import OllamaCloudClient
    from ..paths import db_path
    from .render import doctor_report

    config = Config.from_env()
    client = OllamaCloudClient(config)
    if doctor_report(console, client, config, db_path()):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
