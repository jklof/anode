#!/usr/bin/env python3
"""Emit a lyric fill-in template for an ABC cover score.

Reads a score (e.g. a kept SheetSage2 `score-melody.abc`), analyzes the vocal
staff only, and prints one fill-in section per score section: phrase starts,
note budgets (~= singable syllables), breath/hold marks, and mid-section
meter changes. Fill the numbered blanks with new words following
.opencode/skills/yue2-lyrics-prompts/references/cover-refit-method.md, then
render with cot=melody. Stdlib only; run from the repo root:

    python tools/abc_fit_template.py inputs/temple_of_the_king_cover.abc
    python tools/abc_fit_template.py score.abc --voice Ins --out template.txt
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lyric_fit import plan_template, render_template


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Emit a lyric fill-in template for an ABC cover score.")
    parser.add_argument("score", help="ABC score file to analyze")
    parser.add_argument("--voice", default="Vocal",
                        help="V: staff to plan on (default: Vocal)")
    parser.add_argument("--out", default=None,
                        help="Write template to FILE instead of stdout")
    args = parser.parse_args()
    try:
        abc_text = Path(args.score).read_text(encoding="utf-8")
    except OSError as e:
        print(f"cannot read score: {e}", file=sys.stderr)
        return 1
    rendered = render_template(plan_template(abc_text, voice=args.voice))
    if args.out:
        try:
            Path(args.out).write_text(rendered, encoding="utf-8")
        except OSError as e:
            print(f"cannot write output: {e}", file=sys.stderr)
            return 1
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
