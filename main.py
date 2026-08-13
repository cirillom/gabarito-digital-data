"""Generate missing exam data and rebuild the repository-wide data file."""

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


def main() -> None:
    print("Scanning repository for exams...\n")
    directories = find_exam_directories()

    if not directories:
        print("No directories containing prova.pdf and gabarito.pdf were found.")
        return

    print(f"Found {len(directories)} exam director{'y' if len(directories) == 1 else 'ies'}:\n")
    existing_directories = [directory for directory in directories if (directory / "data.json").exists()]
    missing_directories = [directory for directory in directories if not (directory / "data.json").exists()]
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

    for directory in missing_directories:
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


if __name__ == "__main__":
    main()
