"""OAB-only rich-content extraction and validation command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from main import ROOT_DIR, validate_complete_rich_data
from rich_pipeline import DEFAULT_RICH_WORKERS, MAX_RICH_WORKERS, enrich_rich_data_file


def _exam_directory(edition: int) -> Path:
    directory = ROOT_DIR / "OAB" / "provas" / str(edition)
    if not directory.is_dir():
        raise ValueError(f"OAB {edition} directory does not exist: {directory}")
    return directory


def validate_oab_edition(edition: int) -> list[str]:
    directory = _exam_directory(edition)
    data_path = directory / "data.json"
    pdf_path = directory / "prova.pdf"
    if not data_path.is_file():
        return ["data.json is missing"]
    if not pdf_path.is_file():
        return ["prova.pdf is missing"]
    data = json.loads(data_path.read_text(encoding="utf-8"))
    return validate_complete_rich_data(data, pdf_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume or validate rich extraction for one OAB edition."
    )
    parser.add_argument("edition", type=int, help="OAB edition, for example 46.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate this OAB edition without making Gemini requests.",
    )
    parser.add_argument(
        "--use-gemini",
        action="store_true",
        help="Enhance questions with Gemini vision; valid completed questions are skipped.",
    )
    parser.add_argument(
        "--question",
        type=int,
        action="append",
        help="Limit extraction to one or more question numbers.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_RICH_WORKERS,
        choices=range(1, MAX_RICH_WORKERS + 1),
        metavar="1-4",
        help=f"Maximum concurrent rich requests (default: {DEFAULT_RICH_WORKERS}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate selected questions instead of reusing valid rich content.",
    )
    args = parser.parse_args()

    directory = _exam_directory(args.edition)
    if args.check:
        errors = validate_oab_edition(args.edition)
        if errors:
            print(f"OAB {args.edition}: {len(errors)} rich-content issue(s)")
            for error in errors:
                print(f"  - {error}")
            raise SystemExit(1)
        print(f"OAB {args.edition}: rich content is complete and valid.")
        return

    data = enrich_rich_data_file(
        directory / "data.json",
        directory / "prova.pdf",
        repository_root=ROOT_DIR,
        question_numbers=set(args.question) if args.question else None,
        use_gemini=args.use_gemini,
        force=args.force,
        max_workers=args.workers,
        write=True,
    )
    metadata = data["rich_extraction"]
    print(
        f"OAB {args.edition}: {metadata['successful_question_count']}/"
        f"{metadata['question_count']} rich questions valid ({metadata['status']})."
    )
    if metadata.get("failures"):
        print("Failed questions:")
        for number, error in metadata["failures"].items():
            print(f"  {number}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
