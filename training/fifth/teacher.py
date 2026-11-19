"""Franok 0.5 teacher run: the fourth-round generator and graders with the fifth-round seeds, prompt and grading notes.

Command line: the fourth-round flags plus `--tasks NAME [NAME ...]` (families from seeds.FAMILIES, e.g. `--tasks create code math`), which draws the
seeds only from those families. Without it the default ten families run exactly as before.
"""

import argparse
import sys

from ..fourth import teacher
from . import seeds

GRADING_NOTES = (
    "Judge only the assistant replies; the user may type with typos, chat shorthand or without Turkish letters. "
    "A reply that says plainly it does not know or cannot check today's date, live news, private facts or obscure trivia is correct, "
    "and so is a reply that apologises briefly and asks one clarifying question after the user said the previous reply made no sense or missed the point. "
    "Four-line poems, riddles with their answer and one to four plain-text lines of Python code (code words inside a Turkish reply included) are acceptable when the user asked for them, "
    "but wrong arithmetic or code that does not do what its sentence says still makes the chat BAD."
)


def with_notes(template):
    marker = "Answer with one word"
    if marker not in template:
        raise ValueError("grading template changed: no place to add the fifth-round notes")
    return template.replace(marker, GRADING_NOTES + "\n" + marker, 1)


teacher.build_seeds, teacher.build_prompt = seeds.build_seeds, seeds.build_prompt
teacher.GRADE, teacher.GRADE_STRICT = with_notes(teacher.GRADE), with_notes(teacher.GRADE_STRICT)


def main():
    """Take `--tasks` off the command line before the fourth-round parser (and its vLLM import) sees it."""
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--tasks", nargs="+", choices=sorted(seeds.FAMILIES))
    known, sys.argv[1:] = pre.parse_known_args()
    if known.tasks:
        teacher.build_seeds = lambda *args, **kwargs: seeds.build_seeds(*args, tasks=known.tasks, **kwargs)
    teacher.main()


if __name__ == "__main__":
    main()
