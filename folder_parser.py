"""Build the repository-wide exam data document from per-exam data files."""

import argparse
import json
import os
from pathlib import Path


IGNORED_DIRECTORIES = {".git", ".jj", ".venv", ".vscode", "__pycache__"}


def create_gabarito_json(root_dir: str | Path = ".") -> dict:
    """Return nested data from every ``data.json`` beneath ``root_dir``.

    Each directory maps to a nested object. Invalid files are skipped with a
    warning so one malformed exam file does not prevent the remaining data
    from being collected.
    """
    root = Path(root_dir).resolve()
    final_json_data: dict = {}

    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [name for name in dirnames if name not in IGNORED_DIRECTORIES]
        current_path = Path(dirpath)
        if current_path == root or "data.json" not in filenames:
            continue

        current_level = final_json_data
        for part in current_path.relative_to(root).parts:
            current_level = current_level.setdefault(part, {})

        data_file_path = current_path / "data.json"
        try:
            with data_file_path.open("r", encoding="utf-8") as data_file:
                content = json.load(data_file)
            if not isinstance(content, dict):
                raise ValueError("the JSON root is not an object")
            current_level.update(content)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(f"Warning: Could not read {data_file_path}: {error}")

    return final_json_data


def write_gabarito_json(root_dir: str | Path = ".", output_file: str | Path = "data.json") -> dict:
    """Create the aggregate data and write it to ``output_file``."""
    data = create_gabarito_json(root_dir)
    output_path = Path(output_file)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")
    return data


def main() -> None:
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(description="Aggregate exam data.json files.")
    parser.add_argument("--root-dir", type=Path, default=Path("."), help="Directory to parse.")
    parser.add_argument("--output", type=Path, default=Path("data.json"), help="Output JSON file.")
    args = parser.parse_args()

    data = write_gabarito_json(args.root_dir, args.output)
    print(f"Successfully parsed the directory ({len(data)} top-level entries).")
    print(f"Output saved to {args.output}")


if __name__ == "__main__":
    main()
