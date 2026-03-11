"""
Redline Diff Viewer — generates side-by-side or inline contract redlines.

Uses difflib for text diffing and renders colored HTML diffs in the
style of traditional legal redlines:
  - Red strikethrough = deleted text
  - Green underline = inserted text
  - Highlighted sections = changed clauses
"""

import difflib
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class DiffResult:
    """Result of diffing two contract versions."""
    original_text: str
    revised_text: str
    inline_html: str            # Single-column with insertions/deletions
    side_by_side_html: str      # Two-column diff
    change_summary: dict        # Stats: added_lines, removed_lines, changed_chars
    changed_sections: list[dict]  # List of changed clause sections


class RedlineDiff:
    """
    Generates legal-style redline diffs between contract versions.

    Usage:
        differ = RedlineDiff()
        result = differ.diff(original_text, revised_text)
        # Render result.side_by_side_html in your UI
    """

    def diff(self, original: str, revised: str) -> DiffResult:
        """
        Generate redline diff between two contract texts.

        Args:
            original: Original contract text.
            revised: Revised/amended contract text.

        Returns:
            DiffResult with rendered HTML diffs.
        """
        orig_lines = original.splitlines(keepends=True)
        rev_lines = revised.splitlines(keepends=True)

        inline_html = self._render_inline_diff(orig_lines, rev_lines)
        side_by_side_html = self._render_side_by_side(orig_lines, rev_lines)
        change_summary = self._compute_summary(orig_lines, rev_lines, original, revised)
        changed_sections = self._find_changed_sections(orig_lines, rev_lines)

        return DiffResult(
            original_text=original,
            revised_text=revised,
            inline_html=inline_html,
            side_by_side_html=side_by_side_html,
            change_summary=change_summary,
            changed_sections=changed_sections,
        )

    def _render_inline_diff(self, orig_lines: list[str], rev_lines: list[str]) -> str:
        """Render word-level inline diff."""
        differ = difflib.Differ()
        orig_words = " ".join(orig_lines).split()
        rev_words = " ".join(rev_lines).split()

        diff = list(difflib.ndiff(orig_words, rev_words))

        html_parts = [
            '<div style="font-family: \'IBM Plex Mono\', monospace; font-size: 11.5px; line-height: 1.8; '
            'padding: 18px 20px; background: var(--bg-card); color: var(--ink-mid); border: 1px solid var(--border); border-radius: 4px;">'
        ]

        for item in diff:
            code = item[:2]
            word = item[2:]

            if code == "  ":  # Unchanged
                html_parts.append(f"{word} ")
            elif code == "- ":  # Deleted
                html_parts.append(
                    f'<del style="color: var(--critical); background: var(--critical-bg); '
                    f'text-decoration: line-through; padding: 1px 3px; '
                    f'border-radius: 2px;">{word}</del> '
                )
            elif code == "+ ":  # Inserted
                html_parts.append(
                    f'<ins style="color: var(--low); background: var(--low-bg); '
                    f'text-decoration: none; border-bottom: 2px solid var(--low-bdr); padding: 1px 3px; '
                    f'border-radius: 2px;">{word}</ins> '
                )
            # Skip "? " lines from Differ

        html_parts.append("</div>")
        return "".join(html_parts)

    def _render_side_by_side(self, orig_lines: list[str], rev_lines: list[str]) -> str:
        """Render side-by-side line diff in a two-column layout."""
        matcher = difflib.SequenceMatcher(None, orig_lines, rev_lines)
        opcodes = matcher.get_opcodes()

        left_parts = []
        right_parts = []

        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "equal":
                for line in orig_lines[i1:i2]:
                    text = self._escape_html(line.rstrip())
                    left_parts.append(f'<div class="line equal">{text}</div>')
                    right_parts.append(f'<div class="line equal">{text}</div>')

            elif tag == "replace":
                orig_block = orig_lines[i1:i2]
                rev_block = rev_lines[j1:j2]

                # Word-level diff within replaced block
                orig_text = "".join(orig_block)
                rev_text = "".join(rev_block)

                orig_highlighted = self._word_diff_left(orig_text, rev_text)
                rev_highlighted = self._word_diff_right(orig_text, rev_text)

                left_parts.append(
                    f'<div class="line deleted">{orig_highlighted}</div>'
                )
                right_parts.append(
                    f'<div class="line inserted">{rev_highlighted}</div>'
                )

            elif tag == "delete":
                for line in orig_lines[i1:i2]:
                    text = self._escape_html(line.rstrip())
                    left_parts.append(f'<div class="line deleted">{text}</div>')
                    right_parts.append('<div class="line empty">&nbsp;</div>')

            elif tag == "insert":
                for line in rev_lines[j1:j2]:
                    text = self._escape_html(line.rstrip())
                    left_parts.append('<div class="line empty">&nbsp;</div>')
                    right_parts.append(f'<div class="line inserted">{text}</div>')

        css = """
        <style>
        .diff-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 11.5px;
            line-height: 1.8;
            border: 1px solid var(--border);
            border-radius: 4px;
            overflow: hidden;
            background: var(--bg);
            color: var(--ink-mid);
        }
        .diff-header {
            padding: 10px 16px;
            font-family: 'IBM Plex Mono', monospace;
            font-weight: 500;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            border-bottom: 1px solid var(--border) !important;
        }
        .diff-header.original { background: var(--bg-soft); color: var(--ink-mid); }
        .diff-header.revised  { background: var(--bg-soft); color: var(--ink-mid); border-left: 1px solid var(--border); }
        .diff-col { overflow-y: auto; max-height: 480px; }
        .line { padding: 4px 16px; white-space: pre-wrap; word-break: break-word; }
        .line.equal    { background: transparent; }
        .line.deleted  { background: var(--critical-bg); border-left: 3px solid var(--critical); color: var(--critical); }
        .line.inserted { background: var(--low-bg); border-left: 3px solid var(--low); color: var(--low); }
        .line.empty    { background: var(--bg-soft); }
        .del-word { background: var(--critical-bdr); text-decoration: line-through; border-radius: 2px; padding: 0 2px; color: var(--critical); }
        .ins-word { background: var(--low-bdr); border-radius: 2px; padding: 0 2px; color: var(--low); }
        </style>
        """

        html = css + """
        <div class="diff-container">
            <div class="diff-header original">📄 Original</div>
            <div class="diff-header revised">✏️ Revised</div>
            <div class="diff-col">""" + "".join(left_parts) + """</div>
            <div class="diff-col">""" + "".join(right_parts) + """</div>
        </div>
        """

        return html

    def _word_diff_left(self, orig: str, rev: str) -> str:
        """Highlight deleted words in original."""
        orig_words = orig.split()
        rev_words = rev.split()
        matcher = difflib.SequenceMatcher(None, orig_words, rev_words)
        result = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            chunk = " ".join(orig_words[i1:i2])
            if tag == "equal":
                result.append(self._escape_html(chunk))
            elif tag in ("replace", "delete"):
                result.append(f'<span class="del-word">{self._escape_html(chunk)}</span>')
        return " ".join(result)

    def _word_diff_right(self, orig: str, rev: str) -> str:
        """Highlight inserted words in revised."""
        orig_words = orig.split()
        rev_words = rev.split()
        matcher = difflib.SequenceMatcher(None, orig_words, rev_words)
        result = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            chunk = " ".join(rev_words[j1:j2])
            if tag == "equal":
                result.append(self._escape_html(chunk))
            elif tag in ("replace", "insert"):
                result.append(f'<span class="ins-word">{self._escape_html(chunk)}</span>')
        return " ".join(result)

    def _escape_html(self, text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    def _compute_summary(
        self, orig_lines: list[str], rev_lines: list[str],
        original: str, revised: str
    ) -> dict:
        """Compute change statistics."""
        matcher = difflib.SequenceMatcher(None, orig_lines, rev_lines)
        added = deleted = unchanged = 0

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                unchanged += i2 - i1
            elif tag == "insert":
                added += j2 - j1
            elif tag == "delete":
                deleted += i2 - i1
            elif tag == "replace":
                deleted += i2 - i1
                added += j2 - j1

        orig_chars = len(original)
        rev_chars = len(revised)

        return {
            "added_lines": added,
            "removed_lines": deleted,
            "unchanged_lines": unchanged,
            "original_chars": orig_chars,
            "revised_chars": rev_chars,
            "char_delta": rev_chars - orig_chars,
            "similarity_ratio": round(matcher.ratio(), 3),
            "pct_changed": round((1 - matcher.ratio()) * 100, 1),
        }

    def _find_changed_sections(
        self, orig_lines: list[str], rev_lines: list[str]
    ) -> list[dict]:
        """Find sections (headed by all-caps or numbered headers) that changed."""
        matcher = difflib.SequenceMatcher(None, orig_lines, rev_lines)
        changed = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                # Find nearest heading above this line
                heading = "Unknown Section"
                for idx in range(i1, -1, -1):
                    if idx < len(orig_lines):
                        line = orig_lines[idx].strip()
                        if re.match(r"^(\d+[\.\)]|[A-Z][A-Z\s]{3,})", line):
                            heading = line[:80]
                            break

                changed.append({
                    "tag": tag,
                    "heading": heading,
                    "orig_lines": i1,
                    "rev_lines": j1,
                    "orig_text": "".join(orig_lines[i1:i2])[:200],
                    "rev_text": "".join(rev_lines[j1:j2])[:200],
                })

        return changed
