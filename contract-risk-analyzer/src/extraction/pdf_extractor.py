"""
PDF Extractor — uses PyMuPDF (fitz) to extract text from legal contract PDFs.
Handles multi-column layouts, headers/footers, and page numbers.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import logging; logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    page_num: int
    raw_text: str
    cleaned_text: str
    blocks: list[dict] = field(default_factory=list)


@dataclass
class ExtractedDocument:
    filename: str
    total_pages: int
    raw_text: str
    cleaned_text: str
    pages: list[PageContent] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class PDFExtractor:
    """
    Extracts structured text from PDF contracts using PyMuPDF.

    Usage:
        extractor = PDFExtractor()
        doc = extractor.extract("contract.pdf")
        print(doc.cleaned_text)
    """

    # Patterns to strip from legal docs
    HEADER_FOOTER_PATTERNS = [
        r"^\s*\d+\s*$",                          # Lone page numbers
        r"^\s*Page \d+ of \d+\s*$",              # "Page X of Y"
        r"^\s*CONFIDENTIAL\s*$",                  # Confidentiality stamps
        r"^\s*DRAFT\s*$",                         # Draft watermarks
        r"^\s*-\s*\d+\s*-\s*$",                  # "- 5 -" style page numbers
    ]

    def __init__(self):
        self._header_footer_re = re.compile(
            "|".join(self.HEADER_FOOTER_PATTERNS),
            re.IGNORECASE | re.MULTILINE,
        )

    def extract(self, pdf_path: str | Path) -> ExtractedDocument:
        """
        Extract text from a PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            ExtractedDocument with full text and per-page content.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF is required: pip install PyMuPDF"
            )

        pdf_path = Path(pdf_path)
        logger.info(f"Extracting PDF: {pdf_path.name}")

        doc = fitz.open(str(pdf_path))
        pages: list[PageContent] = []

        for page_num, page in enumerate(doc, start=1):
            raw_text = page.get_text("text")
            blocks_raw = page.get_text("blocks")

            # Build structured blocks (x0, y0, x1, y1, text, block_no, block_type)
            blocks = [
                {
                    "x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3],
                    "text": b[4].strip(),
                    "block_no": b[5],
                    "block_type": b[6],  # 0 = text, 1 = image
                }
                for b in blocks_raw
                if b[6] == 0 and b[4].strip()  # text blocks only
            ]

            cleaned = self._clean_text(raw_text)
            pages.append(PageContent(
                page_num=page_num,
                raw_text=raw_text,
                cleaned_text=cleaned,
                blocks=blocks,
            ))

        metadata = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "creator": doc.metadata.get("creator", ""),
            "page_count": doc.page_count,
        }
        doc.close()

        full_raw = "\n\n".join(p.raw_text for p in pages)
        full_cleaned = "\n\n".join(p.cleaned_text for p in pages)

        logger.info(
            f"Extracted {len(pages)} pages, "
            f"{len(full_cleaned):,} chars from {pdf_path.name}"
        )

        return ExtractedDocument(
            filename=pdf_path.name,
            total_pages=len(pages),
            raw_text=full_raw,
            cleaned_text=full_cleaned,
            pages=pages,
            metadata=metadata,
        )

    def extract_from_bytes(self, pdf_bytes: bytes, filename: str = "upload.pdf") -> ExtractedDocument:
        """
        Extract text directly from PDF bytes (e.g. Streamlit file upload).

        Args:
            pdf_bytes: Raw PDF bytes.
            filename: Logical filename for logging.

        Returns:
            ExtractedDocument
        """
        try:
            import fitz
        except ImportError:
            raise ImportError("PyMuPDF is required: pip install PyMuPDF")

        logger.info(f"Extracting PDF from bytes: {filename}")

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages: list[PageContent] = []

        for page_num, page in enumerate(doc, start=1):
            raw_text = page.get_text("text")
            blocks_raw = page.get_text("blocks")
            blocks = [
                {
                    "x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3],
                    "text": b[4].strip(),
                    "block_no": b[5],
                    "block_type": b[6],
                }
                for b in blocks_raw
                if b[6] == 0 and b[4].strip()
            ]
            cleaned = self._clean_text(raw_text)
            pages.append(PageContent(
                page_num=page_num,
                raw_text=raw_text,
                cleaned_text=cleaned,
                blocks=blocks,
            ))

        metadata = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "page_count": doc.page_count,
        }
        doc.close()

        full_raw = "\n\n".join(p.raw_text for p in pages)
        full_cleaned = "\n\n".join(p.cleaned_text for p in pages)

        return ExtractedDocument(
            filename=filename,
            total_pages=len(pages),
            raw_text=full_raw,
            cleaned_text=full_cleaned,
            pages=pages,
            metadata=metadata,
        )

    def _clean_text(self, text: str) -> str:
        """Remove headers, footers, extra whitespace from page text."""
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            # Remove header/footer patterns
            if self._header_footer_re.match(line):
                continue
            cleaned_lines.append(line)

        cleaned = "\n".join(cleaned_lines)

        # Normalize whitespace while preserving paragraph breaks
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)  # max 2 newlines
        cleaned = re.sub(r"[ \t]+", " ", cleaned)       # collapse spaces
        cleaned = re.sub(r" \n", "\n", cleaned)         # trailing spaces

        return cleaned.strip()
