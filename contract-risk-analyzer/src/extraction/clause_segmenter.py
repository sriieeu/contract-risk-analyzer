"""
Clause Segmenter — uses spaCy + regex heuristics to split contract text
into individual clauses for downstream classification.

Strategy:
1. Detect numbered/lettered clause headers (e.g. "1.", "1.1", "ARTICLE II", "(a)")
2. Detect all-caps section titles
3. Fall back to sentence-level segmentation via spaCy for unnumbered text
"""

import re
from dataclasses import dataclass, field
from typing import Optional

import logging; logger = logging.getLogger(__name__)


@dataclass
class Clause:
    """A single extracted clause from a contract."""
    id: str                     # e.g. "clause_003"
    raw_text: str               # Original text
    heading: Optional[str]      # Detected heading (e.g. "15. Indemnification")
    section_number: Optional[str]  # e.g. "15", "1.2.3"
    char_start: int             # Start offset in document
    char_end: int               # End offset in document
    page_hint: Optional[int]    # Page number if available
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.raw_text.split())


@dataclass
class SegmentationResult:
    clauses: list[Clause]
    total_clauses: int
    method_used: str            # "numbered" | "heading" | "sentence" | "hybrid"
    warnings: list[str] = field(default_factory=list)


class ClauseSegmenter:
    """
    Segments contract text into clauses using a multi-strategy pipeline.

    Usage:
        segmenter = ClauseSegmenter()
        result = segmenter.segment(document_text)
        for clause in result.clauses:
            print(clause.heading, clause.raw_text[:100])
    """

    # Match numbered sections: "1.", "1.1", "1.1.1", "(a)", "SECTION 1", "ARTICLE II"
    NUMBERED_HEADING = re.compile(
        r"""
        (?:^|\n)                          # Start or newline
        (?P<heading>
            (?:SECTION|ARTICLE|CLAUSE)\s+[\dIVXivx]+\.?\s+  # "SECTION 5" / "ARTICLE III"
            |
            \d+(?:\.\d+)*\.?\s+           # "1." / "1.1." / "1.2.3."
            |
            \([a-zA-Z]\)\s+              # "(a)" "(b)"
            |
            [A-Z][A-Z\s]{4,}(?:\n|$)     # ALL-CAPS headings (5+ chars)
        )
        """,
        re.VERBOSE | re.MULTILINE,
    )

    # All-caps standalone heading (alternative pattern)
    ALLCAPS_HEADING = re.compile(
        r"(?:^|\n)([A-Z][A-Z\s\-]{4,})(?:\n|$)",
        re.MULTILINE,
    )

    MIN_CLAUSE_WORDS = 10        # Skip trivially short segments
    MAX_CLAUSE_CHARS = 8000      # Split very long clauses

    def __init__(self, use_spacy: bool = True):
        self.use_spacy = use_spacy
        self._nlp = None
        if use_spacy:
            self._load_spacy()

    def _load_spacy(self):
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
            # Disable components we don't need for speed
            self._nlp.disable_pipes(
                [p for p in self._nlp.pipe_names if p not in ("tok2vec", "senter")]
            )
            logger.info("spaCy loaded: en_core_web_sm")
        except (ImportError, OSError):
            logger.warning(
                "spaCy not available or model not downloaded. "
                "Run: python -m spacy download en_core_web_sm"
            )
            self._nlp = None

    def segment(self, text: str, source_pages: Optional[list] = None) -> SegmentationResult:
        """
        Segment contract text into clauses.

        Args:
            text: Full contract text.
            source_pages: Optional list of (page_num, text) tuples for page hints.

        Returns:
            SegmentationResult with list of Clause objects.
        """
        text = text.strip()
        if not text:
            return SegmentationResult(clauses=[], total_clauses=0, method_used="none")

        # Try numbered segmentation first
        numbered_clauses = self._segment_numbered(text)

        if len(numbered_clauses) >= 3:
            method = "numbered"
            raw_clauses = numbered_clauses
        else:
            # Fall back to paragraph/sentence segmentation
            method = "paragraph"
            raw_clauses = self._segment_paragraphs(text)

        # Filter short segments
        filtered = [c for c in raw_clauses if c.word_count >= self.MIN_CLAUSE_WORDS]

        # Assign page hints if available
        if source_pages:
            filtered = self._assign_pages(filtered, source_pages)

        warnings = []
        if len(filtered) == 0:
            warnings.append("No clauses detected. Contract may be image-based (OCR required).")
        elif len(filtered) < 5:
            warnings.append(f"Only {len(filtered)} clauses detected. Contract may use non-standard formatting.")

        logger.info(f"Segmented into {len(filtered)} clauses via '{method}' method")

        return SegmentationResult(
            clauses=filtered,
            total_clauses=len(filtered),
            method_used=method,
            warnings=warnings,
        )

    def _segment_numbered(self, text: str) -> list[Clause]:
        """Split on numbered/heading patterns."""
        matches = list(self.NUMBERED_HEADING.finditer(text))
        if not matches:
            return []

        clauses = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            clause_text = text[start:end].strip()
            heading_text = match.group("heading").strip() if match.group("heading") else None

            # Extract section number
            section_num = None
            if heading_text:
                num_match = re.match(r"^(\d+(?:\.\d+)*)", heading_text)
                if num_match:
                    section_num = num_match.group(1)

            clauses.append(Clause(
                id=f"clause_{i+1:03d}",
                raw_text=clause_text,
                heading=heading_text,
                section_number=section_num,
                char_start=start,
                char_end=end,
                page_hint=None,
            ))

        return clauses

    def _segment_paragraphs(self, text: str) -> list[Clause]:
        """Split on double newlines (paragraph boundaries)."""
        paragraphs = re.split(r"\n\s*\n", text)
        clauses = []

        for i, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue

            # Try to detect if first line is a heading
            lines = para.split("\n")
            heading = None
            if len(lines) > 1 and len(lines[0]) < 100:
                first_line = lines[0].strip()
                if first_line.isupper() or re.match(r"^\d+[\.\)]\s+\w", first_line):
                    heading = first_line

            clauses.append(Clause(
                id=f"clause_{i+1:03d}",
                raw_text=para,
                heading=heading,
                section_number=None,
                char_start=text.find(para),
                char_end=text.find(para) + len(para),
                page_hint=None,
            ))

        return clauses

    def _assign_pages(
        self, clauses: list[Clause], source_pages: list
    ) -> list[Clause]:
        """
        Assign approximate page numbers to clauses based on char offsets.
        source_pages: list of PageContent objects with .cleaned_text
        """
        # Build cumulative char offset map
        offset = 0
        page_offsets = []
        for page in source_pages:
            page_offsets.append((page.page_num, offset, offset + len(page.cleaned_text)))
            offset += len(page.cleaned_text) + 2  # +2 for \n\n separator

        for clause in clauses:
            for page_num, start, end in page_offsets:
                if start <= clause.char_start < end:
                    clause.page_hint = page_num
                    break

        return clauses
