import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from catalog_db import connect, open_catalog, replace_exam
from main import partition_exam_directories, run
from pdf_parser import (
    build_prompt,
    normalize_answer_keys,
    normalize_exam_variant,
    parse_exam_directory,
)
from rich_content import QuotaExceededError


def _exam_data() -> dict:
    return {
        "data": "2026-01-01",
        "variante": "Caderno 1 - Azul",
        "qtd_questoes": 1,
        "opcoes_resposta": ["A", "B"],
        "disciplinas": ["Teste"],
        "questoes": {"1": {"disciplina": "Teste", "resposta": "A"}},
    }


class GeneratorTests(unittest.TestCase):
    def test_repository_generation_stops_at_first_quota_error(self) -> None:
        class QuotaError(Exception):
            code = 429

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            directories = [
                root / "ENEM" / "provas" / "2025",
                root / "ENEM" / "provas" / "2026",
            ]
            for directory in directories:
                directory.mkdir(parents=True)
            parser = MagicMock(side_effect=QuotaError("quota exhausted"))

            with (
                patch("main.ROOT_DIR", root),
                patch("main.LOG_DIR", root / "logs"),
                patch("main.parse_exam_directory", parser),
            ):
                with self.assertRaises(QuotaExceededError):
                    run(
                        directories=directories,
                        database_path=root / "catalog.sqlite3",
                        rich_content=False,
                    )
            self.assertEqual(parser.call_count, 1)

    def test_default_generation_only_sends_missing_database_exam_to_ai(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            existing = root / "ENEM" / "provas" / "2025"
            missing = root / "ENEM" / "provas" / "2026"
            existing.mkdir(parents=True)
            missing.mkdir(parents=True)
            connection = open_catalog(root / "catalog.sqlite3")
            replace_exam(connection, existing, _exam_data(), repository_root=root)

            refresh, generate = partition_exam_directories(
                connection,
                [existing, missing],
                regenerate_all=False,
                repository_root=root,
            )
            connection.close()

            self.assertEqual(refresh, [existing])
            self.assertEqual(generate, [missing])

    def test_regenerate_all_sends_every_exam_to_ai(self) -> None:
        directories = [Path("first"), Path("second")]
        connection = MagicMock()

        refresh, generate = partition_exam_directories(
            connection, directories, regenerate_all=True
        )

        self.assertEqual(refresh, [])
        self.assertEqual(generate, directories)

    def test_prompt_requires_official_subject_research(self) -> None:
        prompt = build_prompt()
        self.assertIn("research the official subject list", prompt)
        self.assertIn('"disciplinas"', prompt)
        self.assertIn("Every questoes[*].disciplina value", prompt)
        self.assertIn("OAB questions must use the legal disciplines", prompt)

    def test_prompt_requires_the_exact_exam_variant(self) -> None:
        prompt = build_prompt()
        self.assertIn('"variante": "Caderno 7 - Azul"', prompt)
        self.assertIn("distinguishing booklet color", prompt)
        self.assertIn("Do not invent a color or version", prompt)

    def test_exam_variant_is_required_and_trimmed(self) -> None:
        data = {"variante": "  Tipo 1 - Branca  "}
        normalize_exam_variant(data)
        self.assertEqual(data["variante"], "Tipo 1 - Branca")
        with self.assertRaisesRegex(ValueError, "exact exam booklet"):
            normalize_exam_variant({})

    def test_normalizes_annulments_and_rejects_malformed_answers(self) -> None:
        data = {
            "opcoes_resposta": ["A", "B", "C", "D"],
            "questoes": {
                "1": {"resposta": "anulada"},
                "2": {"resposta": " b "},
            },
        }
        normalize_answer_keys(data)
        self.assertEqual(data["questoes"]["1"]["resposta"], "N/A")
        self.assertEqual(data["questoes"]["2"]["resposta"], "B")
        data["questoes"]["3"] = {"resposta": "Aidão"}
        with self.assertRaisesRegex(ValueError, "neither a selectable option"):
            normalize_answer_keys(data)

    def test_oab_answer_data_matches_official_annulments(self) -> None:
        connection = connect(Path(__file__).resolve().parent / "catalog.sqlite3", read_only=True)
        rows = connection.execute(
            "SELECT exam.title, question.number FROM question "
            "JOIN exam ON exam.id = question.exam_id "
            "JOIN institution ON institution.id = exam.institution_id "
            "WHERE institution.name = 'OAB' AND question.answer = 'N/A'"
        ).fetchall()
        connection.close()
        actual: dict[int, set[int]] = {}
        for row in rows:
            edition = int(row["title"].split(" | ", 1)[0])
            actual.setdefault(edition, set()).add(row["number"])
        self.assertEqual(actual.get(32), {3, 45, 55, 61, 74})
        self.assertEqual(actual.get(33), {59})

    def test_parser_uses_google_genai_client_without_writing_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            (directory / "prova.pdf").write_bytes(b"test prova")
            (directory / "gabarito.pdf").write_bytes(b"test gabarito")
            client = MagicMock()
            client.files.upload.side_effect = [
                SimpleNamespace(
                    name="files/prova",
                    state=SimpleNamespace(name="ACTIVE"),
                    display_name="prova.pdf",
                ),
                SimpleNamespace(
                    name="files/gabarito",
                    state=SimpleNamespace(name="ACTIVE"),
                    display_name="gabarito.pdf",
                ),
            ]
            client.models.generate_content.return_value = SimpleNamespace(
                text=json.dumps(_exam_data())
            )

            with (
                patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}),
                patch("pdf_parser.genai.Client", return_value=client),
                patch("pdf_parser.apply_layout_to_data", return_value=True),
            ):
                data = parse_exam_directory(directory)

            self.assertEqual(client.files.upload.call_count, 2)
            client.models.generate_content.assert_called_once()
            self.assertEqual(data["questoes"]["1"]["resposta"], "A")
            self.assertFalse((directory / "data.json").exists())


if __name__ == "__main__":
    unittest.main()
