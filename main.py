"""Generate exam data and rebuild the repository-wide data file."""

import argparse
from pathlib import Path

from folder_parser import write_gabarito_json
from pdf_parser import parse_exam_directory
from question_layout import enrich_data_file
from rich_content import enrich_rich_data_file, write_html_preview


ROOT_DIR = Path(__file__).resolve().parent
MAIN_DATA = ROOT_DIR / "data.json"


def find_exam_directories(root_dir: Path = ROOT_DIR) -> list[Path]:
    """Find directories containing both required exam PDFs."""
    return sorted(
        prova.parent
        for prova in root_dir.rglob("prova.pdf")
        if (prova.parent / "gabarito.pdf").is_file()
    )


def partition_exam_directories(
    directories: list[Path], *, regenerate_all: bool
) -> tuple[list[Path], list[Path]]:
    """Return directories to refresh locally and directories to regenerate with AI."""
    if regenerate_all:
        return [], directories
    return (
        [directory for directory in directories if (directory / "data.json").exists()],
        [directory for directory in directories if not (directory / "data.json").exists()],
    )


def run(
    *,
    regenerate_all: bool = False,
    rich_content: bool = True,
    use_gemini_rich: bool = False,
    rich_questions: set[int] | None = None,
    preview_dir: Path | None = None,
) -> None:
    print("Scanning repository for exams...\n")
    directories = find_exam_directories()

    if not directories:
        print("No directories containing prova.pdf and gabarito.pdf were found.")
        return

    print(f"Found {len(directories)} exam director{'y' if len(directories) == 1 else 'ies'}:\n")
    existing_directories, generation_directories = partition_exam_directories(
        directories, regenerate_all=regenerate_all
    )
    failures: list[tuple[Path, str]] = []

    # Refresh every existing exam before attempting network-backed Gemini work.
    # This keeps layout backfills deterministic even when one new exam fails.
    for directory in existing_directories:
        relative_directory = directory.relative_to(ROOT_DIR)
        data_file = directory / "data.json"
        print(f"{relative_directory}")
        print("  Extracting question layout from prova.pdf...")
        try:
            succeeded = enrich_data_file(data_file, directory / "prova.pdf")
            print(
                "  Layout extraction "
                f"{'succeeded' if succeeded else 'failed; PDF mode disabled'}"
            )
            if not succeeded:
                failures.append((relative_directory, "layout extraction failed"))
        except (OSError, ValueError) as error:
            failures.append((relative_directory, str(error)))
            print(f"  Failed: {error}")

    for directory in generation_directories:
        relative_directory = directory.relative_to(ROOT_DIR)
        print(f"{relative_directory}")
        print("  Generating data.json...")
        try:
            parse_exam_directory(directory, repository_root=ROOT_DIR)
        except Exception as error:
            # Keep processing so one network/API failure cannot prevent the
            # deterministic aggregate from being rebuilt for valid exams.
            failures.append((relative_directory, str(error)))
            print(f"  Failed: {error}")

    if rich_content:
        print("\nExtracting rich question content...")
        for directory in directories:
            relative_directory = directory.relative_to(ROOT_DIR)
            data_file = directory / "data.json"
            if not data_file.is_file():
                continue
            print(f"{relative_directory}")
            try:
                enriched = enrich_rich_data_file(
                    data_file,
                    directory / "prova.pdf",
                    repository_root=ROOT_DIR,
                    question_numbers=rich_questions,
                    use_gemini=use_gemini_rich,
                    force=regenerate_all,
                )
                metadata = enriched["rich_extraction"]
                print(
                    "  Rich extraction "
                    f"{metadata['successful_question_count']}/"
                    f"{metadata['question_count']} ({metadata['status']})"
                )
                if preview_dir is not None:
                    preview_path = preview_dir / relative_directory / "index.html"
                    write_html_preview(
                        enriched,
                        directory=directory,
                        output_path=preview_path,
                        question_numbers=rich_questions,
                    )
                    print(f"  Preview: {preview_path}")
            except (OSError, ValueError, RuntimeError) as error:
                failures.append((relative_directory, f"rich extraction: {error}"))
                print(f"  Rich extraction failed: {error}")

    write_gabarito_json(ROOT_DIR, MAIN_DATA)
    print(f"\nUpdated {MAIN_DATA.relative_to(ROOT_DIR)}")
    if failures:
        print("\nCompleted with failures:")
        for directory, error in failures:
            print(f"  {directory}: {error}")
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate exam JSON files and rebuild the aggregate catalog."
    )
    parser.add_argument(
        "--regenerate-all",
        action="store_true",
        help="Regenerate every exam data.json with Gemini, even when it already exists.",
    )
    parser.add_argument(
        "--rich-content",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build rich text, image, formula, and option content from local PDFs (default: enabled).",
    )
    parser.add_argument(
        "--use-gemini-rich",
        action="store_true",
        help="Use Gemini vision to enhance rich content (requires --rich-content).",
    )
    parser.add_argument(
        "--rich-question",
        type=int,
        action="append",
        help="Limit rich extraction to one or more question numbers.",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        help="Write local HTML previews beneath this directory.",
    )
    args = parser.parse_args()
    if args.use_gemini_rich and not args.rich_content:
        parser.error("--use-gemini-rich requires --rich-content")
    if args.rich_question and not args.rich_content:
        parser.error("--rich-question requires --rich-content")
    run(
        regenerate_all=args.regenerate_all,
        rich_content=args.rich_content,
        use_gemini_rich=args.use_gemini_rich,
        rich_questions=set(args.rich_question) if args.rich_question else None,
        preview_dir=args.preview_dir.resolve() if args.preview_dir else None,
    )


if __name__ == "__main__":
    main()
