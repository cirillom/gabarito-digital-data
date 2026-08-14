"""Generate exam data and rebuild the repository-wide data file."""

import argparse
from pathlib import Path

from folder_parser import write_gabarito_json
from pdf_parser import parse_exam_directory
from question_layout import enrich_data_file


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


def run(*, regenerate_all: bool = False) -> None:
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
    args = parser.parse_args()
    run(regenerate_all=args.regenerate_all)


if __name__ == "__main__":
    main()
