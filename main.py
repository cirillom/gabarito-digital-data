"""Generate missing exam data and rebuild the repository-wide data file."""

from pathlib import Path

from folder_parser import write_gabarito_json
from pdf_parser import parse_exam_directory


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
    for directory in directories:
        relative_directory = directory.relative_to(ROOT_DIR)
        data_file = directory / "data.json"
        print(f"{relative_directory}")

        if data_file.exists():
            print("  data.json already exists")
            continue

        print("  Generating data.json...")
        parse_exam_directory(directory, repository_root=ROOT_DIR)

    write_gabarito_json(ROOT_DIR, MAIN_DATA)
    print(f"\nUpdated {MAIN_DATA.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
