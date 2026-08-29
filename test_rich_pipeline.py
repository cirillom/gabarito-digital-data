import hashlib
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pymupdf

from rich_pipeline import (
    _RetryingGeminiClient,
    _RetryingOpenAIClient,
    _StopRequested,
    enrich_rich_data_file,
)
from rich_content import QuotaExceededError


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
    def test_quota_error_is_not_retried(self) -> None:
        class QuotaError(Exception):
            code = 429

        models = MagicMock()
        models.generate_content.side_effect = QuotaError("quota exhausted")
        client = _RetryingGeminiClient(
            SimpleNamespace(models=models),
            question_number=17,
            max_attempts=5,
        )

        with self.assertRaises(QuotaExceededError):
            client.models.generate_content(model="test", contents=[])

        self.assertEqual(models.generate_content.call_count, 1)

    def test_openai_quota_error_is_not_retried(self) -> None:
        class QuotaError(Exception):
            status_code = 429

        responses = MagicMock()
        responses.parse.side_effect = QuotaError("rate limit")
        client = _RetryingOpenAIClient(
            SimpleNamespace(responses=responses),
            question_number=17,
            max_attempts=5,
        )

        with self.assertRaises(QuotaExceededError):
            client.responses.parse(model="test", input=[])

        self.assertEqual(responses.parse.call_count, 1)

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

    def test_stop_request_prevents_another_retry(self) -> None:
        class TemporaryError(Exception):
            code = 503

        stop_event = threading.Event()
        models = MagicMock()
        models.generate_content.side_effect = TemporaryError("busy")

        def request_stop(_: str) -> None:
            stop_event.set()

        client = _RetryingGeminiClient(
            SimpleNamespace(models=models),
            question_number=17,
            max_attempts=5,
            base_delay=10,
            max_delay=20,
            stop_event=stop_event,
            status_callback=request_stop,
        )

        with patch("rich_pipeline.random.uniform", return_value=0.0):
            with self.assertRaises(_StopRequested):
                client.models.generate_content(model="test", contents=[])

        self.assertEqual(models.generate_content.call_count, 1)

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
                    enrich_rich_data_file(
                        data_path,
                        pdf_path,
                        write=True,
                        max_workers=1,
                        progress=False,
                    )

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
                    progress=False,
                )

            rich = result["questoes"]["1"]["conteudo"]["rich"]
            self.assertEqual(rich["method"], "deterministic")
            self.assertIn("1", result["rich_extraction"]["failures"])

    def test_openai_provider_is_forwarded_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            pdf_path = directory / "prova.pdf"
            data_path = directory / "data.json"
            _write_pdf(pdf_path)
            _write_data(data_path, question_count=1)
            digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
            rich = _rich(digest, method="openai:gpt-5.6-luna")

            with (
                patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
                patch("openai.OpenAI", return_value=MagicMock()),
                patch(
                    "rich_pipeline.extract_question_rich_content",
                    return_value=rich,
                ) as extractor,
            ):
                result = enrich_rich_data_file(
                    data_path,
                    pdf_path,
                    provider="openai",
                    write=False,
                    max_workers=1,
                    progress=False,
                )

            self.assertEqual(result["rich_extraction"]["method"], "openai:gpt-5.6-luna")
            self.assertEqual(extractor.call_args.kwargs["provider"], "openai")
            self.assertIsNotNone(extractor.call_args.kwargs["openai_client"])

    def test_quota_error_stops_before_the_next_question(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            pdf_path = directory / "prova.pdf"
            data_path = directory / "data.json"
            _write_pdf(pdf_path)
            _write_data(data_path)
            extractor = MagicMock(side_effect=QuotaExceededError("quota exhausted"))

            with (
                patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}),
                patch("google.genai.Client", return_value=MagicMock()),
                patch(
                    "rich_pipeline.extract_question_rich_content",
                    extractor,
                ),
            ):
                with self.assertRaises(QuotaExceededError):
                    enrich_rich_data_file(
                        data_path,
                        pdf_path,
                        use_gemini=True,
                        write=True,
                        max_workers=1,
                        progress=False,
                    )

            self.assertEqual(extractor.call_count, 1)
            checkpoint = json.loads(data_path.read_text(encoding="utf-8"))
            self.assertEqual(
                checkpoint["rich_extraction"]["processed_question_count"], 0
            )


if __name__ == "__main__":
    unittest.main()
