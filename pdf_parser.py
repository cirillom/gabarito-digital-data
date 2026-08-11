"""Extract exam data from a pair of PDF files using Gemini.

The public :func:`parse_exam_directory` function can be imported by other
Python code. Run this file directly to use the command-line interface.
"""

import argparse
import json
import os
from pathlib import Path
from urllib.parse import quote

import google.generativeai as genai
from dotenv import load_dotenv


PDF_FILENAMES = ("prova.pdf", "gabarito.pdf")
MODEL_NAME = "gemini-3.5-flash-lite"


def build_pdf_link(directory: Path, repository_root: Path) -> str:
    """Return the GitHub raw URL for the exam's ``prova.pdf`` file."""
    relative_path = directory.resolve().relative_to(repository_root.resolve()) / "prova.pdf"
    return (
        "https://raw.githubusercontent.com/cirillom/gabarito-digital-data/"
        "refs/heads/main/"
        f"{quote(relative_path.as_posix())}"
    )


def extract_json(response_text: str) -> dict:
    """Parse Gemini output, accepting either raw JSON or a fenced JSON block."""
    text = response_text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            text = "\n".join(lines[1:-1]).strip()

    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("Gemini response must be a JSON object.")
    return result


def build_prompt() -> str:
    """Build the instruction sent alongside the uploaded PDFs."""
    return """
You are a specialized data extraction API. Your sole function is to process two uploaded PDF files, prova.pdf and gabarito.pdf, and generate a single, precise JSON output.

Constraints:
  - The output MUST be only the raw JSON object.

Instructions:
 - Analyze the prova.pdf to identify the specific exam version (e.g., "Prova V", "Prova K", etc.).
 - Using the identified exam version, locate the corresponding answer key column in the gabarito.pdf.
 - Read both documents to extract all necessary information.
 - If a piece of information is not explicitly available in the documents, perform a web search to find the correct data.
 - Populate the following JSON structure exactly as specified.

JSON Output Structure:
{
    "data": "2024-01-01",
    "qtd_questoes": 2,
    "opcoes_resposta": ["A", "B", "C", "D", "E"],
    "questoes": {
        "1": {"disciplina": "Matemática", "resposta": "A"},
        "2": {"disciplina": "História", "resposta": "B"}
    }
}

Field Population Rules:
  - data: Extract the exam date from the documents and format it as YYYY-MM-DD.
  - qtd_questoes: Determine the total count of questions in the exam.
  - opcoes_resposta: This field should be a static array: ["A", "B", "C", "D", "E"].
  - questoes: This must be an object containing entries for every question number (from 1 to the total). For each question:
      - disciplina: Determine the academic discipline based on the question's content in prova.pdf. Use "Interdisciplinar" if it blends multiple distinct fields.
      - resposta: Extract the correct single-letter answer from the matched answer key in gabarito.pdf. Invalid, annulled, or non-existent answers should be represented as "N/A".
""".strip()


def parse_exam_directory(
    directory: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> dict:
    """Generate and save ``data.json`` for an exam directory.

    The directory must contain ``prova.pdf`` and ``gabarito.pdf``. The parsed
    dictionary, including its ``pdf_link``, is returned after it is saved.
    """
    directory = Path(directory).resolve()
    repository_root = Path(repository_root or Path(__file__).resolve().parent).resolve()

    missing_files = [name for name in PDF_FILENAMES if not (directory / name).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing required file(s) in {directory}: {', '.join(missing_files)}"
        )

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to the environment or .env file.")

    genai.configure(api_key=api_key)
    uploaded_files = []
    for filename in PDF_FILENAMES:
        file_path = directory / filename
        print(f"Uploading {file_path}...")
        uploaded_files.append(genai.upload_file(path=str(file_path), display_name=str(file_path)))

    print("Sending prompt to Gemini...")
    model = genai.GenerativeModel(model_name=MODEL_NAME)
    response = model.generate_content([build_prompt(), *uploaded_files])
    data = extract_json(response.text)
    data["pdf_link"] = build_pdf_link(directory, repository_root)

    output_path = directory / "data.json"
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=False)

    return data


def main() -> None:
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(description="Extract exam data from PDF files.")
    parser.add_argument("--directory", "-d", type=Path, required=True, help="Exam directory.")
    args = parser.parse_args()

    try:
        parse_exam_directory(args.directory)
    except (FileNotFoundError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        parser.exit(1, f"Error: {error}\n")
    except Exception as error:
        parser.exit(1, f"Unexpected error while parsing PDFs: {error}\n")


if __name__ == "__main__":
    main()
