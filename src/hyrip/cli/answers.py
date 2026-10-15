"""Answer grammar shared by the CLI loop.

Value answers: a letter, a roman numeral, a paragraph letter, or free text — parsed by the
grader. Navigation takes precedence: ``p``/``\\`` back, ``s`` skip, ``f`` finish, ``v`` view
the article again, ``l`` re-list headings, ``q`` quit-saving, ``?`` help. ``b`` stays
reserved for TFNG letters.
"""

from __future__ import annotations

from dataclasses import dataclass

NAV_COMMANDS = {"p", "\\", "s", "f", "v", "l", "q", "?"}
HELP_TEXT = (
    "answer grammar: type your answer (letter, word(s), or phrase)\n"
    "  p or \\  go back one question        v  re-read the article (no clock pause)\n"
    "  s       skip this question          l  show the heading list again\n"
    "  f       finish and mark the paper   q  quit (session saved as abandoned)\n"
    "  ?       this help"
)


@dataclass
class Action:
    kind: str  # "value" | "back" | "skip" | "finish" | "view" | "list" | "quit" | "help"
    raw: str | None = None


def parse_answer(line: str) -> Action:
    token = line.strip()
    if not token:
        return Action(kind="value", raw="")  # empty = re-prompt
    lowered = token.lower()
    if lowered in NAV_COMMANDS:
        mapping = {
            "p": "back",
            "\\": "back",
            "s": "skip",
            "f": "finish",
            "v": "view",
            "l": "list",
            "q": "quit",
            "?": "help",
        }
        return Action(kind=mapping[lowered])
    return Action(kind="value", raw=token)
