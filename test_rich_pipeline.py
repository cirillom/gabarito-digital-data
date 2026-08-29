import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pymupdf

from rich_pipeline import _RetryingGeminiClient, enrich_rich_data_file


def _write_pdf(path: Path) -> None:
    document = pymupdf.open()
    document.new_page(width=200, height=200)
    document.save(path)
    document.close()


def _rich(pdf_sha256: str, *, method: str = "deterministic") -> dict:
    def document(text: str) -> dict:
        return {
            "blocks": [
                {
                    "type": "paragraph",
                    "inlines": [{"type": "text", "text": text}],
                    "source": [{"page": 1, "rect": [0.0, 0.0, 1.0, 1.0]}],
                }
            ]
        }

    return {
        "version": 2,
        "status": "success",
        "source_pdf_sha256": pdf_sha256,
        "method": method,
        "statement": document("Statement"),
        "options": [
            {"label": label, "content": document(f"Option {label}")}
            for label in ["A", "B"]
        ],
        "assets": [],
    }


def _write_data(path: Path, question_count: int = 2) -> None:
    path.write_text(
        json.dumps(
            {
                "qtd_questoes": question_count,
                "opcoes_resposta": ["A", "B"],
                "questoes": {
                    str(number): {
                        "resposta": "A",
                        "disciplina": "Test",
                        "conteudo": {
                            "segments": [
                                {
                                    "page": 1,
                                    "rect": [0.0, 0.0, 1.0, 1.0],
                                    "kind": "question",
                                }
                            ]
                        },
                    }
                    for number in range(1, question_count + 1)
                },
            }
        ),
        encoding="utf-8",
    )


class RichPipelineTest(unittest.TestCase):
    def test_retries_503_with_bounded_backoff(self) -> None:
        class TemporaryError(Exception):
            code = 503

        models = MagicMock()
        models.generate_content.side_effect = [
            TemporaryError("busy"),
            TemporaryError("still busy"),
            SimpleNamespace(text="ok"),
        ]
        client = _RetryingGeminiClient(
            SimpleNamespace(models=models),
            question_number=17,
            max_attempts=3,
            base_delay=0.01,
            max_delay=0.02,
        )

        with (
            patch("rich_pipeline.time.sleep") as sleep,
            patch("rich_pipeline.random.uniform", return_value=0.0),
        ):
            response = client.models.generate_content(model="test", contents=[])

        self.assertEqual(response.text, "ok")
        self.assertEqual(models.generate_content.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_successful_question_is_checkpointed_before_later_interrupt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            pdf_path = directory / "prova.pdf"
            data_path = directory / "data.json"
            _write_pdf(pdf_path)
            _write_data(data_path)
            digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()

            with patch(
                "rich_pipeline.extract_question_rich_content",
                side_effect=[_rich(digest), KeyboardInterrupt()],
            ):
                with self.assertRaises(KeyboardInterrupt):
                    enrich_rich_data_file(data_path, pdf_path, write=True, max_workers=1)

            checkpoint = json.loads(data_path.read_text(encoding="utf-8"))
            self.assertEqual(
                checkpoint["questoes"]["1"]["conteudo"]["rich"]["status"],
                "success",
            )
            self.assertNotIn("rich", checkpoint["questoes"]["2"]["conteudo"])
            self.assertEqual(
                checkpoint["rich_extraction"]["successful_question_count"], 1
            )

    def test_failed_gemini_upgrade_preserves_valid_deterministic_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            pdf_path = directory / "prova.pdf"
            data_path = directory / "data.json"
            _write_pdf(pdf_path)
            _write_data(data_path, question_count=1)
            digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
            data = json.loads(data_path.read_text(encoding="utf-8"))
            data["questoes"]["1"]["conteudo"]["rich"] = _rich(digest)
            data_path.write_text(json.dumps(data), encoding="utf-8")

            with (
                patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}),
                patch("google.genai.Client", return_value=MagicMock()),
                patch(
                    "rich_pipeline.extract_question_rich_content",
                    side_effect=RuntimeError("Gemini unavailable"),
                ),
            ):
                result = enrich_rich_data_file(
                    data_path,
                    pdf_path,
                    use_gemini=True,
                    write=True,
                    max_workers=1,
                )

            rich = result["questoes"]["1"]["conteudo"]["rich"]
            self.assertEqual(rich["method"], "deterministic")
            self.assertIn("1", result["rich_extraction"]["failures"])


if __name__ == "__main__":
    unittest.main()
