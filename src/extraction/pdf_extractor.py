"""
Section-aware PDF text extractor using PyMuPDF.

Handles two-column academic paper layouts, detects section boundaries,
and strips the references section from the main text.

Usage:
    extractor = PDFExtractor(config)
    result = extractor.extract(paper_record)
"""

import re
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import ftfy
from loguru import logger

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("Install PyMuPDF: pip install pymupdf")


# Section header patterns (matched case-insensitively)
SECTION_PATTERNS = {
    "abstract": re.compile(r"^abstract$", re.IGNORECASE),
    "introduction": re.compile(r"^(1\.?\s*)?introduction$", re.IGNORECASE),
    "background": re.compile(r"^(2\.?\s*)?background$", re.IGNORECASE),
    "related_work": re.compile(r"^related\s+work$", re.IGNORECASE),
    "methods": re.compile(
        r"^(materials?\s+and\s+methods?|methods?|methodology|experimental\s+procedures?|"
        r"\d+\.?\s*(materials?\s+and\s+)?methods?)$",
        re.IGNORECASE,
    ),
    "results": re.compile(r"^(\d+\.?\s*)?results?$", re.IGNORECASE),
    "discussion": re.compile(r"^(\d+\.?\s*)?discussion$", re.IGNORECASE),
    "conclusion": re.compile(r"^(\d+\.?\s*)?conclusions?$", re.IGNORECASE),
    "summary": re.compile(r"^summary$", re.IGNORECASE),
    "acknowledgments": re.compile(r"^(acknowledgments?|acknowledgements?)$", re.IGNORECASE),
    "references": re.compile(r"^references?$", re.IGNORECASE),
    "supplementary": re.compile(r"^(supplementary|supplemental)\s*(material|methods|information)?$", re.IGNORECASE),
}

REFERENCES_SECTION_KEY = "references"


@dataclass
class ExtractionResult:
    paper_id: int
    pdf_path: str
    sections: dict[str, str]   # section_name -> text
    full_text: str              # concatenated non-references text
    extraction_quality: float   # 0.0 – 1.0 heuristic score
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "pdf_path": self.pdf_path,
            "sections": self.sections,
            "full_text": self.full_text,
            "extraction_quality": self.extraction_quality,
            "error": self.error,
        }


@dataclass
class ExtractorConfig:
    timeout_seconds: int = 60
    min_text_chars: int = 500
    max_section_chars: int = 8000
    max_fulltext_chars: int = 40000
    strip_references: bool = True


class PDFExtractor:
    def __init__(self, config: Optional[ExtractorConfig] = None):
        self.config = config or ExtractorConfig()

    def extract(self, paper_id: int, pdf_path: Path) -> ExtractionResult:
        """Extract structured text from a single PDF. Returns an ExtractionResult."""
        start = time.time()
        pdf_path_str = str(pdf_path)

        try:
            # Use SIGALRM on Unix for timeout enforcement
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(self.config.timeout_seconds)

            result = self._extract_impl(paper_id, pdf_path)
            signal.alarm(0)  # cancel alarm

            elapsed = time.time() - start
            logger.debug(f"Extracted {pdf_path.name} in {elapsed:.1f}s — quality={result.extraction_quality:.2f}")
            return result

        except TimeoutError:
            signal.alarm(0)
            logger.warning(f"Timeout ({self.config.timeout_seconds}s) extracting {pdf_path.name}")
            return ExtractionResult(
                paper_id=paper_id,
                pdf_path=pdf_path_str,
                sections={},
                full_text="",
                extraction_quality=0.0,
                error="timeout",
            )
        except Exception as exc:
            signal.alarm(0)
            logger.warning(f"Error extracting {pdf_path.name}: {exc}")
            return ExtractionResult(
                paper_id=paper_id,
                pdf_path=pdf_path_str,
                sections={},
                full_text="",
                extraction_quality=0.0,
                error=str(exc),
            )

    def _extract_impl(self, paper_id: int, pdf_path: Path) -> ExtractionResult:
        doc = fitz.open(str(pdf_path))
        all_blocks = self._extract_blocks(doc)
        doc.close()

        raw_text = "\n".join(b["text"] for b in all_blocks)
        raw_text = _clean_text(raw_text)

        if len(raw_text) < self.config.min_text_chars:
            return ExtractionResult(
                paper_id=paper_id,
                pdf_path=str(pdf_path),
                sections={},
                full_text=raw_text,
                extraction_quality=0.1,
                error="too_short",
            )

        sections = self._segment_sections(all_blocks)
        sections = self._truncate_sections(sections)

        # Build full_text from non-references sections
        excluded = {REFERENCES_SECTION_KEY, "acknowledgments", "supplementary"}
        body_parts = []
        for key in ["abstract", "introduction", "background", "methods", "results", "discussion", "conclusion"]:
            if key in sections and sections[key]:
                body_parts.append(sections[key])
        # Fallback: any remaining sections not excluded
        if not body_parts:
            body_parts = [v for k, v in sections.items() if k not in excluded and v]

        if not body_parts:
            body_parts = [raw_text]

        full_text = "\n\n".join(body_parts)[: self.config.max_fulltext_chars]

        quality = _compute_quality(sections, full_text)

        return ExtractionResult(
            paper_id=paper_id,
            pdf_path=str(pdf_path),
            sections=sections,
            full_text=full_text,
            extraction_quality=quality,
        )

    def _extract_blocks(self, doc: "fitz.Document") -> list[dict]:
        """
        Extract text blocks from all pages, handling two-column layouts.
        Returns list of dicts with keys: text, page, x0, y0, x1, y1, is_header.
        """
        all_blocks = []
        for page_num, page in enumerate(doc):
            page_width = page.rect.width
            raw_blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)

            text_blocks = [b for b in raw_blocks if b[6] == 0 and b[4].strip()]

            # Detect two-column layout by checking x-center distribution
            two_column = _is_two_column(text_blocks, page_width)

            if two_column:
                # Sort: left column first (x0 < midpoint), then right column, each top-to-bottom
                midpoint = page_width / 2
                left = sorted([b for b in text_blocks if b[0] < midpoint], key=lambda b: b[1])
                right = sorted([b for b in text_blocks if b[0] >= midpoint], key=lambda b: b[1])
                ordered_blocks = left + right
            else:
                ordered_blocks = sorted(text_blocks, key=lambda b: (b[1], b[0]))

            for b in ordered_blocks:
                text = _clean_text(b[4])
                if not text:
                    continue
                all_blocks.append({
                    "text": text,
                    "page": page_num,
                    "x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3],
                    "is_header": _is_section_header(text),
                })

        return all_blocks

    def _segment_sections(self, blocks: list[dict]) -> dict[str, str]:
        """
        Walk through blocks and assign text to sections based on detected headers.
        Returns {section_key: text}.
        """
        sections: dict[str, list[str]] = {}
        current_section: Optional[str] = None
        past_references = False

        for block in blocks:
            text = block["text"]

            if block["is_header"]:
                section_key = _classify_header(text)
                if section_key:
                    current_section = section_key
                    if section_key == REFERENCES_SECTION_KEY:
                        past_references = True
                    if section_key not in sections:
                        sections[section_key] = []
                    continue  # Don't include the header text itself

            if self.config.strip_references and past_references and current_section == REFERENCES_SECTION_KEY:
                continue  # Skip reference list text

            if current_section:
                sections.setdefault(current_section, []).append(text)

        return {k: "\n".join(v).strip() for k, v in sections.items()}

    def _truncate_sections(self, sections: dict[str, str]) -> dict[str, str]:
        return {k: v[: self.config.max_section_chars] for k, v in sections.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_two_column(blocks: list, page_width: float) -> bool:
    """Heuristic: check if x-centers of blocks cluster in left and right halves."""
    if len(blocks) < 6:
        return False
    x_centers = [(b[0] + b[2]) / 2 / page_width for b in blocks]
    left_count = sum(1 for x in x_centers if x < 0.5)
    right_count = len(x_centers) - left_count
    # Both halves have substantial content → two-column
    return left_count >= 3 and right_count >= 3


def _is_section_header(text: str) -> bool:
    """Heuristic: a header is short (≤ 60 chars), mostly on one line, and matches a known pattern."""
    stripped = text.strip()
    if len(stripped) > 70 or "\n" in stripped.strip():
        return False
    return _classify_header(stripped) is not None


def _classify_header(text: str) -> Optional[str]:
    """Return the section key if text matches a known section header, else None."""
    stripped = text.strip()
    for key, pattern in SECTION_PATTERNS.items():
        if pattern.match(stripped):
            return key
    return None


def _clean_text(text: str) -> str:
    """Fix Unicode issues, normalize whitespace, strip control chars."""
    text = ftfy.fix_text(text)
    # Collapse multiple blank lines to one
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip form feeds and carriage returns
    text = text.replace("\r", "").replace("\f", "\n")
    return text.strip()


def _compute_quality(sections: dict[str, str], full_text: str) -> float:
    """
    Heuristic quality score 0.0 – 1.0 based on:
    - Whether key sections were detected
    - Total text length
    """
    score = 0.0
    if len(full_text) > 2000:
        score += 0.3
    if len(full_text) > 5000:
        score += 0.2
    if "methods" in sections and len(sections["methods"]) > 200:
        score += 0.2
    if "results" in sections and len(sections["results"]) > 200:
        score += 0.2
    if "abstract" in sections and len(sections["abstract"]) > 50:
        score += 0.1
    return min(score, 1.0)


def _timeout_handler(signum, frame):
    raise TimeoutError("PDF extraction timed out")
