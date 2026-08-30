"""Detect question regions in exam PDFs.

The extractor deliberately keeps the PDF as the visual source of truth.  It
stores normalized page rectangles instead of trying to recreate text,
equations, tables, or images.  A layout is only marked as successful when an
ordered anchor was found for every question in an exam.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pymupdf


LAYOUT_VERSION = 1
ENGINE_NAME = "pymupdf"
ENGINE_REVISION = 6
ANCHOR_PADDING = 4.0
MIN_SEGMENT_HEIGHT = 6.0


class LayoutExtractionError(ValueError):
    """Raised when a PDF cannot be segmented with sufficient confidence."""


@dataclass(frozen=True)
class Boundary:
    page: int  # zero based
    column: int
    bbox: tuple[float, float, float, float]
    kind: str
    number: int | None = None
    shared_range: tuple[int, int] | None = None

    @property
    def lane(self) -> int:
        return self.page * 2 + self.column

    @property
    def sort_key(self) -> tuple[int, int, float, float]:
        return (self.page, self.column, self.bbox[1], self.bbox[0])


@dataclass(frozen=True)
class DocumentProfile:
    name: str
    body_top_ratio: float
    body_bottom_ratio: float


_ENEM_PROFILE = DocumentProfile("enem", 0.075, 0.945)
_FUVEST_PROFILE = DocumentProfile("fuvest", 0.040, 0.962)
_OAB_PROFILE = DocumentProfile("oab", 0.050, 0.918)
_GENERIC_PROFILE = DocumentProfile("generic", 0.024, 0.956)

_ENEM_ANCHOR = re.compile(r"^QUESTAO\s*0*(\d{1,3})\b")
_NUMBER_ANCHOR = re.compile(r"^0*(\d{1,3})(?:[.)])?(?:\s|$)")
_FUVEST_SHARED = re.compile(
    r"\bTEXTOS? PARA AS QUESTOES(?: DE)?\s+(\d{1,3})\s+(?:E|A)\s+(\d{1,3})\b"
)
_TJSP_SHARED = re.compile(
    r"\bLEIA O TEXTO PARA RESPONDER (?:A|AS) QUESTOES DE NUMEROS?\s+"
    r"(\d{1,3})\s+A\s+(\d{1,3})\b"
)
_ENCODED_DIGITS = str.maketrans("ϬϭϮϯϰϱϲϳϴϵ", "0123456789")


def _normalize_text(value: str) -> str:
    decoded_digits = value.translate(_ENCODED_DIGITS)
    decomposed = unicodedata.normalize("NFKD", decoded_digits)
    without_accents = "".join(
        " " if unicodedata.category(character) == "Cc" else character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", without_accents).strip().upper()


def _is_bold(span: dict[str, Any]) -> bool:
    font_name = str(span.get("font", "")).lower()
    return bool(int(span.get("flags", 0)) & 16) or any(
        marker in font_name for marker in ("bold", "black", "heavy", "demi")
    )


def _union_bbox(items: Iterable[Iterable[float]]) -> tuple[float, float, float, float]:
    rectangles = [tuple(float(value) for value in item) for item in items]
    if not rectangles:
        raise LayoutExtractionError("Cannot create a bounding box from no items.")
    return (
        min(item[0] for item in rectangles),
        min(item[1] for item in rectangles),
        max(item[2] for item in rectangles),
        max(item[3] for item in rectangles),
    )


def _column_for_bbox(bbox: tuple[float, float, float, float], page_width: float) -> int:
    return 0 if (bbox[0] + bbox[2]) / 2 < page_width / 2 else 1


def _iter_text_lines(page: pymupdf.Page) -> Iterable[tuple[str, list[dict[str, Any]], tuple]]:
    page_dict = page.get_text("dict")
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = [span for span in line.get("spans", []) if span.get("text", "").strip()]
            if not spans:
                continue
            text = " ".join(str(span["text"]).strip() for span in spans)
            yield text, spans, tuple(float(value) for value in line["bbox"])


def _is_english_language_section(text: str) -> bool:
    normalized = _normalize_text(text)
    return (
        "LINGUA INGLESA" in normalized
        or "OPCAO INGLES" in normalized
        or "OPCAO: INGLES" in normalized
    )


def _is_spanish_language_section(text: str) -> bool:
    normalized = _normalize_text(text)
    return (
        "LINGUA ESPANHOLA" in normalized
        or "OPCAO ESPANHOL" in normalized
        or "OPCAO: ESPANHOL" in normalized
    )


def _language_section_starts(document: pymupdf.Document) -> tuple[int | None, int | None]:
    english_start: int | None = None
    spanish_start: int | None = None
    for page_index, page in enumerate(document):
        text = page.get_text("text")
        if english_start is None and _is_english_language_section(text):
            english_start = page_index
        if spanish_start is None and _is_spanish_language_section(text):
            spanish_start = page_index
    return english_start, spanish_start


def _find_anchor_candidates(
    document: pymupdf.Document,
    expected_numbers: set[int],
) -> tuple[list[Boundary], DocumentProfile]:
    enem_candidates: list[Boundary] = []
    generic_candidates: list[tuple[Boundary, float, bool]] = []

    for page_index, page in enumerate(document):
        width = float(page.rect.width)
        height = float(page.rect.height)
        normalized_page_text = _normalize_text(page.get_text("text"))
        is_instruction_page = (
            "INFORMACOES GERAIS" in normalized_page_text
            and "NAO SERA PERMITIDO" in normalized_page_text
        )
        for text, spans, line_bbox in _iter_text_lines(page):
            normalized = _normalize_text(text)
            enem_match = _ENEM_ANCHOR.match(normalized)
            if enem_match:
                number = int(enem_match.group(1))
                if number in expected_numbers:
                    bbox = tuple(float(value) for value in line_bbox)
                    enem_candidates.append(
                        Boundary(
                            page=page_index,
                            column=_column_for_bbox(bbox, width),
                            bbox=bbox,
                            kind="anchor_candidate",
                            number=number,
                        )
                    )
                continue

            number_match = _NUMBER_ANCHOR.match(normalized)
            if not number_match or is_instruction_page:
                continue
            number = int(number_match.group(1))
            if number not in expected_numbers:
                continue

            first_span = spans[0]
            if not _is_bold(first_span):
                continue
            bbox = tuple(float(value) for value in first_span["bbox"])
            x0, y0 = bbox[:2]
            at_column_start = x0 < width * 0.18 or width * 0.48 < x0 < width * 0.65
            within_body_band = height * 0.02 < y0 < height * 0.94
            if not at_column_start or not within_body_band:
                continue

            punctuation_style = "." in str(first_span.get("text", ""))
            generic_candidates.append(
                (
                    Boundary(
                        page=page_index,
                        column=_column_for_bbox(bbox, width),
                        bbox=bbox,
                        kind="anchor_candidate",
                        number=number,
                    ),
                    float(first_span.get("size", 0)),
                    punctuation_style,
                )
            )

    if enem_candidates:
        return sorted(enem_candidates, key=lambda item: item.sort_key), _ENEM_PROFILE
    if not generic_candidates:
        raise LayoutExtractionError("No supported question-number anchors were found.")

    candidates = [item[0] for item in generic_candidates]
    sizes = [item[1] for item in generic_candidates]
    uses_number_period = any(item[2] for item in generic_candidates)
    if uses_number_period:
        profile = _GENERIC_PROFILE
    elif max(sizes, default=0) >= 12.5:
        profile = _FUVEST_PROFILE
    else:
        profile = _OAB_PROFILE
    return sorted(candidates, key=lambda item: item.sort_key), profile


def _select_ordered_anchors(
    candidates: list[Boundary],
    expected_numbers: list[int],
) -> list[Boundary]:
    """Select the first complete monotonic path through the candidate stream.

    This intentionally chooses the first ENEM language branch and ignores the
    later OAB survey, both of which repeat otherwise valid question numbers.
    """
    selected: list[Boundary] = []
    expected_index = 0
    for candidate in candidates:
        if expected_index >= len(expected_numbers):
            break
        if candidate.number == expected_numbers[expected_index]:
            selected.append(
                Boundary(
                    page=candidate.page,
                    column=candidate.column,
                    bbox=candidate.bbox,
                    kind="question",
                    number=candidate.number,
                )
            )
            expected_index += 1

    if expected_index != len(expected_numbers):
        missing = expected_numbers[expected_index : expected_index + 5]
        raise LayoutExtractionError(
            "Could not find a complete ordered question sequence; "
            f"next missing number(s): {missing}."
        )
    return selected


def _select_enem_language_branch(
    document: pymupdf.Document,
    candidates: list[Boundary],
    selected: list[Boundary],
    raw_questions: dict[str, Any],
) -> list[Boundary]:
    """Choose the language branch indicated by question metadata when possible."""
    if not set(range(1, 6)).issubset({item.number for item in selected}):
        return selected

    discipline_hints = {
        _normalize_text(str(raw_questions[str(number)].get("disciplina", "")))
        for number in range(1, 6)
        if isinstance(raw_questions.get(str(number)), dict)
    }
    wants_spanish = any("ESPAN" in hint for hint in discipline_hints)
    wants_english = any("INGL" in hint for hint in discipline_hints)
    if not wants_spanish and not wants_english:
        return selected

    english_start, spanish_start = _language_section_starts(document)
    target_start = spanish_start if wants_spanish else english_start
    if target_start is None:
        raise LayoutExtractionError(
            "The requested ENEM language section could not be identified."
        )

    target_pages = set(range(target_start, document.page_count))
    candidate_language_anchors = [
        candidate
        for candidate in candidates
        if candidate.number in range(1, 6) and candidate.page in target_pages
    ]
    # Select the first coherent 1..5 path after the chosen section header. This
    # supports branches that span several pages and excludes the later branch.
    replacements: dict[int, Boundary] = {}
    next_number = 1
    for candidate in candidate_language_anchors:
        if candidate.number == next_number:
            replacements[next_number] = Boundary(
                page=candidate.page,
                column=candidate.column,
                bbox=candidate.bbox,
                kind="question",
                number=next_number,
            )
            next_number += 1
            if next_number == 6:
                break

    if set(replacements) != set(range(1, 6)):
        raise LayoutExtractionError(
            "The requested ENEM language branch does not contain questions 1 through 5."
        )

    replaced = [replacements.get(item.number, item) for item in selected]
    if replaced != sorted(replaced, key=lambda item: item.sort_key):
        raise LayoutExtractionError("The selected ENEM language branch is out of order.")
    return replaced


def _find_shared_passages(
    document: pymupdf.Document,
    expected_numbers: set[int],
) -> list[Boundary]:
    passages: list[Boundary] = []
    for page_index, page in enumerate(document):
        width = float(page.rect.width)
        for block in page.get_text("blocks"):
            normalized = _normalize_text(str(block[4]))
            match = _FUVEST_SHARED.search(normalized) or _TJSP_SHARED.search(normalized)
            if not match:
                continue
            first, last = (int(match.group(1)), int(match.group(2)))
            if first > last:
                first, last = last, first
            referenced = set(range(first, last + 1))
            if not referenced or not referenced.issubset(expected_numbers):
                raise LayoutExtractionError(
                    f"Shared passage references unexpected questions {first} to {last}."
                )
            bbox = tuple(float(value) for value in block[:4])
            passages.append(
                Boundary(
                    page=page_index,
                    column=_column_for_bbox(bbox, width),
                    bbox=bbox,
                    kind="shared",
                    shared_range=(first, last),
                )
            )
    return sorted(passages, key=lambda item: item.sort_key)


def _excluded_pages(document: pymupdf.Document) -> set[int]:
    excluded: set[int] = set()
    for page_index, page in enumerate(document):
        normalized = _normalize_text(page.get_text("text"))
        if "PROPOSTA DE REDACAO" in normalized or "FOLHA DE RASCUNHO" in normalized:
            excluded.add(page_index)
    return excluded


def _body_bounds(page: pymupdf.Page, profile: DocumentProfile) -> tuple[float, float]:
    height = float(page.rect.height)
    return height * profile.body_top_ratio, height * profile.body_bottom_ratio


def _make_segment(
    document: pymupdf.Document,
    *,
    page_index: int,
    column: int,
    top: float,
    bottom: float,
    kind: str,
) -> dict[str, Any] | None:
    page = document[page_index]
    width = float(page.rect.width)
    height = float(page.rect.height)
    top = max(0.0, min(height, top))
    bottom = max(0.0, min(height, bottom))
    if bottom - top < MIN_SEGMENT_HEIGHT:
        return None
    left = 0.0 if column == 0 else width / 2
    right = width / 2 if column == 0 else width
    rect = [
        round(left / width, 6),
        round(top / height, 6),
        round(right / width, 6),
        round(bottom / height, 6),
    ]
    return {"page": page_index + 1, "rect": rect, "kind": kind}


def _segments_between(
    document: pymupdf.Document,
    start: Boundary,
    end: Boundary | None,
    *,
    profile: DocumentProfile,
    excluded_pages: set[int],
    kind: str,
) -> list[dict[str, Any]]:
    start_lane = start.lane
    start_page = start.page

    if end is None:
        # Never absorb trailing survey/instruction pages into the final question.
        end_lane = start.page * 2 + 1
        end_y: float | None = None
    else:
        end_lane = end.lane
        end_y = end.bbox[1] - ANCHOR_PADDING
        crosses_excluded_page = any(
            page in excluded_pages for page in range(start.page + 1, end.page)
        )
        starts_unselected_section = end.kind == "other_anchor" and end.page > start.page
        if crosses_excluded_page or starts_unselected_section:
            end_lane = start.page * 2 + 1
            end_y = None
        elif end.page > start.page + 1:
            raise LayoutExtractionError(
                "A question appears to span more than two PDF pages."
            )

    if end_lane < start_lane:
        raise LayoutExtractionError("A question boundary appears before its anchor.")
    if end_lane - start_lane > 4:
        raise LayoutExtractionError(
            f"A region spans an unreasonable number of page columns ({end_lane - start_lane + 1})."
        )

    segments: list[dict[str, Any]] = []
    for lane in range(start_lane, end_lane + 1):
        page_index, column = divmod(lane, 2)
        if page_index in excluded_pages:
            continue
        page = document[page_index]
        body_top, body_bottom = _body_bounds(page, profile)
        top = start.bbox[1] - ANCHOR_PADDING if lane == start_lane else body_top
        bottom = end_y if lane == end_lane and end_y is not None else body_bottom
        segment = _make_segment(
            document,
            page_index=page_index,
            column=column,
            top=max(body_top, top),
            bottom=min(body_bottom, bottom),
            kind=kind,
        )
        if segment is not None:
            segments.append(segment)
    return segments


def _segment_has_visible_content(
    document: pymupdf.Document,
    segment: dict[str, Any],
) -> bool:
    page = document[int(segment["page"]) - 1]
    left, top, right, bottom = segment["rect"]
    clip = pymupdf.Rect(
        left * page.rect.width,
        top * page.rect.height,
        right * page.rect.width,
        bottom * page.rect.height,
    )
    if page.get_text("text", clip=clip).strip():
        return True
    if any(pymupdf.Rect(item["bbox"]).intersects(clip) for item in page.get_image_info()):
        return True
    return any(pymupdf.Rect(item["rect"]).intersects(clip) for item in page.get_drawings())


def _remove_empty_tail_segments(
    document: pymupdf.Document,
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop only empty continuation lanes; the anchor lane is mandatory."""
    if not segments or not _segment_has_visible_content(document, segments[0]):
        return []
    end = len(segments)
    while end > 1 and not _segment_has_visible_content(document, segments[end - 1]):
        end -= 1
    return segments[:end]


def _validate_segments(
    document: pymupdf.Document,
    layouts: dict[int, list[dict[str, Any]]],
    anchors: dict[int, Boundary],
) -> None:
    if set(layouts) != set(anchors):
        raise LayoutExtractionError("The extracted question set is incomplete.")

    for number, segments in layouts.items():
        own_segments = [segment for segment in segments if segment.get("kind") == "question"]
        if not own_segments:
            raise LayoutExtractionError(f"Question {number} has no question segment.")
        anchor = anchors[number]
        first = own_segments[0]
        if first["page"] != anchor.page + 1:
            raise LayoutExtractionError(f"Question {number}'s first segment is on the wrong page.")

        for segment in segments:
            page_number = segment.get("page")
            rect = segment.get("rect")
            if (
                not isinstance(page_number, int)
                or isinstance(page_number, bool)
                or not 1 <= page_number <= document.page_count
                or segment.get("kind") not in {"question", "shared"}
                or not isinstance(rect, list)
                or len(rect) != 4
            ):
                raise LayoutExtractionError(
                    f"Question {number} has an invalid serialized segment."
                )
            if not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in rect
            ):
                raise LayoutExtractionError(f"Question {number} has non-finite coordinates.")
            left, top, right, bottom = rect
            if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
                raise LayoutExtractionError(
                    f"Question {number} has invalid normalized coordinates."
                )
            if not _segment_has_visible_content(document, segment):
                raise LayoutExtractionError(
                    f"Question {number} has a segment without visible PDF content."
                )

        page = document[anchor.page]
        anchor_x = ((anchor.bbox[0] + anchor.bbox[2]) / 2) / page.rect.width
        anchor_y = ((anchor.bbox[1] + anchor.bbox[3]) / 2) / page.rect.height
        left, top, right, bottom = first["rect"]
        if not (left <= anchor_x <= right and top <= anchor_y <= bottom):
            raise LayoutExtractionError(
                f"Question {number}'s first segment does not contain its anchor."
            )


def _build_layouts_with_profile(
    document: pymupdf.Document,
    *,
    candidates: list[Boundary],
    selected: list[Boundary],
    expected_numbers: set[int],
    profile: DocumentProfile,
) -> dict[int, list[dict[str, Any]]]:
    """Segment and validate an already detected question sequence."""
    selected_keys = {(item.page, item.column, item.bbox, item.number) for item in selected}
    other_candidates = [
        Boundary(
            page=item.page,
            column=item.column,
            bbox=item.bbox,
            kind="other_anchor",
            number=item.number,
        )
        for item in candidates
        if (item.page, item.column, item.bbox, item.number) not in selected_keys
    ]
    shared_passages = _find_shared_passages(document, expected_numbers)
    boundaries = sorted(
        [*selected, *other_candidates, *shared_passages],
        key=lambda item: item.sort_key,
    )
    boundary_index = {id(item): index for index, item in enumerate(boundaries)}
    excluded = _excluded_pages(document)

    layouts: dict[int, list[dict[str, Any]]] = {}
    anchors_by_number = {item.number: item for item in selected if item.number is not None}
    for anchor in selected:
        index = boundary_index[id(anchor)]
        next_boundary = boundaries[index + 1] if index + 1 < len(boundaries) else None
        segments = _segments_between(
            document,
            anchor,
            next_boundary,
            profile=profile,
            excluded_pages=excluded,
            kind="question",
        )
        segments = _remove_empty_tail_segments(document, segments)
        layouts[anchor.number] = segments  # type: ignore[index]

    for passage in shared_passages:
        first, last = passage.shared_range or (0, -1)
        first_anchor = anchors_by_number.get(first)
        if first_anchor is None or passage.sort_key >= first_anchor.sort_key:
            raise LayoutExtractionError(
                f"Shared passage for questions {first}-{last} does not precede question {first}."
            )
        shared_segments = _segments_between(
            document,
            passage,
            first_anchor,
            profile=profile,
            excluded_pages=excluded,
            kind="shared",
        )
        shared_segments = _remove_empty_tail_segments(document, shared_segments)
        if not shared_segments:
            raise LayoutExtractionError(
                f"Shared passage for questions {first}-{last} is empty."
            )
        for number in range(first, last + 1):
            layouts[number] = [*shared_segments, *layouts[number]]

    _validate_segments(document, layouts, anchors_by_number)  # type: ignore[arg-type]
    return layouts


def extract_question_layout(
    pdf_path: str | Path,
    question_numbers: Iterable[int],
    *,
    raw_questions: dict[str, Any] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    """Return ordered PDF segments for every supplied question number."""
    pdf_path = Path(pdf_path)
    expected_numbers = sorted(set(question_numbers))
    if not expected_numbers:
        raise LayoutExtractionError("The exam has no numeric question keys.")

    with pymupdf.open(pdf_path) as document:
        if not document.page_count:
            raise LayoutExtractionError("The PDF has no pages.")
        if not any(page.get_text("text").strip() for page in document):
            raise LayoutExtractionError("The PDF has no usable text layer; OCR is required.")

        candidates, profile = _find_anchor_candidates(document, set(expected_numbers))
        selected = _select_ordered_anchors(candidates, expected_numbers)
        if profile == _ENEM_PROFILE and raw_questions is not None:
            selected = _select_enem_language_branch(
                document,
                candidates,
                selected,
                raw_questions,
            )
        expected_number_set = set(expected_numbers)
        try:
            return _build_layouts_with_profile(
                document,
                candidates=candidates,
                selected=selected,
                expected_numbers=expected_number_set,
                profile=profile,
            )
        except LayoutExtractionError as primary_error:
            if profile == _GENERIC_PROFILE:
                raise
            try:
                return _build_layouts_with_profile(
                    document,
                    candidates=candidates,
                    selected=selected,
                    expected_numbers=expected_number_set,
                    profile=_GENERIC_PROFILE,
                )
            except LayoutExtractionError as fallback_error:
                raise LayoutExtractionError(
                    f"{primary_error} Generic profile retry failed: {fallback_error}"
                ) from fallback_error


def _has_complete_cached_layout(raw_questions: dict[str, Any]) -> bool:
    for question in raw_questions.values():
        if not isinstance(question, dict):
            return False
        content = question.get("conteudo")
        if not isinstance(content, dict):
            return False
        segments = content.get("segments")
        if not isinstance(segments, list) or not segments:
            return False
        has_question_segment = False
        for segment in segments:
            if not isinstance(segment, dict):
                return False
            page_number = segment.get("page")
            rect = segment.get("rect")
            kind = segment.get("kind")
            if (
                not isinstance(page_number, int)
                or isinstance(page_number, bool)
                or page_number < 1
                or kind not in {"question", "shared"}
                or not isinstance(rect, list)
                or len(rect) != 4
                or not all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    for value in rect
                )
            ):
                return False
            left, top, right, bottom = rect
            if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
                return False
            has_question_segment = has_question_segment or kind == "question"
        if not has_question_segment:
            return False
    return True


def _pdf_digest(pdf_path: Path) -> str:
    digest = hashlib.sha256()
    with pdf_path.open("rb") as pdf_file:
        for chunk in iter(lambda: pdf_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_layout_to_data(data: dict[str, Any], pdf_path: str | Path) -> bool:
    """Mutate exam data with layout metadata, returning whether it succeeded."""
    pdf_path = Path(pdf_path)
    raw_questions = data.get("questoes")
    if not isinstance(raw_questions, dict):
        raise LayoutExtractionError('"questoes" must be an object before layout extraction.')

    previous_layout = data.get("layout_extraction")
    try:
        pdf_sha256 = _pdf_digest(pdf_path)
    except OSError as error:
        for raw_question in raw_questions.values():
            if isinstance(raw_question, dict):
                raw_question.pop("conteudo", None)
        data["layout_extraction"] = {
            "status": "failed",
            "version": LAYOUT_VERSION,
            "engine": ENGINE_NAME,
            "engine_revision": ENGINE_REVISION,
            "error": str(error),
        }
        return False

    if (
        isinstance(previous_layout, dict)
        and previous_layout.get("status") == "success"
        and previous_layout.get("version") == LAYOUT_VERSION
        and previous_layout.get("engine_revision") == ENGINE_REVISION
        and previous_layout.get("pdf_sha256") == pdf_sha256
        and previous_layout.get("question_count") == len(raw_questions)
        and _has_complete_cached_layout(raw_questions)
    ):
        return True

    # Never leave stale regions behind after a failed re-extraction.
    for raw_question in raw_questions.values():
        if isinstance(raw_question, dict):
            raw_question.pop("conteudo", None)

    try:
        question_numbers = [int(key) for key in raw_questions]
        expected_count = data.get("qtd_questoes")
        if (
            not isinstance(expected_count, int)
            or isinstance(expected_count, bool)
            or expected_count != len(question_numbers)
        ):
            raise LayoutExtractionError(
                '"qtd_questoes" does not match the number of question entries.'
            )
        if len(set(question_numbers)) != len(question_numbers):
            raise LayoutExtractionError("Question numbers are duplicated.")

        layouts = extract_question_layout(
            pdf_path,
            question_numbers,
            raw_questions=raw_questions,
        )
        for number, segments in layouts.items():
            question = raw_questions.get(str(number))
            if not isinstance(question, dict):
                raise LayoutExtractionError(f"Question {number} is not an object.")
            question["conteudo"] = {"segments": segments}

        data["layout_extraction"] = {
            "status": "success",
            "version": LAYOUT_VERSION,
            "engine": ENGINE_NAME,
            "engine_revision": ENGINE_REVISION,
            "question_count": len(question_numbers),
            "pdf_sha256": pdf_sha256,
        }
        return True
    except (LayoutExtractionError, OSError, RuntimeError, ValueError) as error:
        data["layout_extraction"] = {
            "status": "failed",
            "version": LAYOUT_VERSION,
            "engine": ENGINE_NAME,
            "engine_revision": ENGINE_REVISION,
            "error": str(error),
        }
        return False

