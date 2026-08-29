import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from main import partition_exam_directories, run
from rich_content import QuotaExceededError
import json

from pdf_parser import (
    build_prompt,
    normalize_answer_keys,
    normalize_exam_description,
    parse_exam_directory,
)


class GeneratorTests(unittest.TestCase):
    def test_repository_generation_stops_at_first_quota_error(self) -> None:
        class QuotaError(Exception):
            code = 429

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            directories = [root / "first", root / "second"]
            for directory in directories:
                directory.mkdir()
            parser = MagicMock(side_effect=QuotaError("quota exhausted"))

            with (
                patch("main.ROOT_DIR", root),
                patch("main.MAIN_DATA", root / "data.json"),
                patch("main.parse_exam_directory", parser),
                patch("main.write_gabarito_json") as write_catalog,
            ):
                with self.assertRaises(QuotaExceededError):
                    run(
                        directories=directories,
                        rich_content=False,
                    )

            self.assertEqual(parser.call_count, 1)
            write_catalog.assert_called_once()

    def test_default_generation_only_sends_missing_data_to_ai(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            existing = root / "existing"
            missing = root / "missing"
            existing.mkdir()
            missing.mkdir()
            (existing / "data.json").write_text("{}", encoding="utf-8")

            refresh, generate = partition_exam_directories(
                [existing, missing], regenerate_all=False
            )

            self.assertEqual(refresh, [existing])
            self.assertEqual(generate, [missing])

    def test_regenerate_all_sends_every_exam_to_ai(self) -> None:
        directories = [Path("first"), Path("second")]

        refresh, generate = partition_exam_directories(
            directories, regenerate_all=True
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

        self.assertIn('"descricao": "Caderno 7 - Azul"', prompt)
        self.assertIn("every distinguishing booklet color", prompt)
        self.assertIn("Do not invent a color or version", prompt)

    def test_exam_description_is_required_and_trimmed(self) -> None:
        data = {"descricao": "  Tipo 1 - Branca  "}

        normalize_exam_description(data)

        self.assertEqual(data["descricao"], "Tipo 1 - Branca")
        with self.assertRaisesRegex(ValueError, "exact exam variant"):
            normalize_exam_description({})

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
        root = Path(__file__).resolve().parent
        expected = {
            "32": {3, 45, 55, 61, 74},
            "33": {59},
        }
        for edition, expected_questions in expected.items():
            data = json.loads(
                (root / "OAB" / "provas" / edition / "data.json").read_text(
                    encoding="utf-8"
                )
            )
            invalidated = {
                int(number)
                for number, question in data["questoes"].items()
                if question["resposta"] == "N/A"
            }
            self.assertEqual(invalidated, expected_questions)

    def test_parser_uses_google_genai_client_files_and_models_apis(self) -> None:
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
                text=json.dumps(
                    {
                        "data": "2026-01-01",
                        "descricao": "Versão única",
                        "qtd_questoes": 1,
                        "opcoes_resposta": ["A", "B"],
                        "disciplinas": ["Teste"],
                        "questoes": {
                            "1": {"disciplina": "Teste", "resposta": "A"}
                        },
                    }
                )
            )

            with (
                patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}),
                patch("pdf_parser.genai.Client", return_value=client),
                patch("pdf_parser.apply_layout_to_data", return_value=True),
            ):
                data = parse_exam_directory(
                    directory,
                    repository_root=directory,
                )

            self.assertEqual(client.files.upload.call_count, 2)
            client.models.generate_content.assert_called_once()
            self.assertEqual(data["questoes"]["1"]["resposta"], "A")
            self.assertEqual(data["descricao"], "Versão única")
            self.assertTrue((directory / "data.json").is_file())


if __name__ == "__main__":
    unittest.main()
