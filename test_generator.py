import tempfile
import unittest
from pathlib import Path

from main import partition_exam_directories
from pdf_parser import build_prompt


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


if __name__ == "__main__":
    unittest.main()
