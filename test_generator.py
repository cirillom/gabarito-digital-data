import tempfile
import unittest
from pathlib import Path

from main import partition_exam_directories
import json

from pdf_parser import build_prompt, normalize_answer_keys


class GeneratorTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
