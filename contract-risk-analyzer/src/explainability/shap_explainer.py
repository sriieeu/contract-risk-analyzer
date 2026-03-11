"""
SHAP Explainer — uses SHAP to explain what text features drove each
clause risk classification.

Approach:
- For BERT models: use shap.Explainer with transformers pipeline
- For keyword fallback: use token-level importance based on keyword matches

Produces per-token SHAP values that can be visualized as highlighted text.
"""

from dataclasses import dataclass, field
from typing import Optional

import logging; logger = logging.getLogger(__name__)


@dataclass
class TokenImportance:
    """Importance score for a single token."""
    token: str
    importance: float        # Signed SHAP value
    is_positive: bool        # True = increases risk, False = decreases


@dataclass
class ExplanationResult:
    """SHAP explanation for a clause."""
    clause_id: str
    clause_type: str
    token_importances: list[TokenImportance]
    top_positive_tokens: list[TokenImportance]   # Most risk-increasing
    top_negative_tokens: list[TokenImportance]   # Most risk-decreasing
    explanation_html: str                         # Pre-rendered HTML with highlights
    method: str                                   # "shap_bert" | "keyword_importance"


class SHAPExplainer:
    """
    Explains clause risk classifications using SHAP.

    Usage:
        explainer = SHAPExplainer(model, tokenizer)
        result = explainer.explain(clause_text, predicted_type)
        print(result.explanation_html)
    """

    def __init__(self, model=None, tokenizer=None):
        self._model = model
        self._tokenizer = tokenizer
        self._shap_explainer = None
        self._initialized = False

    def initialize(self, pipeline=None) -> bool:
        """
        Initialize SHAP explainer with a HuggingFace pipeline.

        Returns True if SHAP is available, False if falling back to keyword method.
        """
        try:
            import shap
            if pipeline is not None:
                self._shap_explainer = shap.Explainer(pipeline)
                self._initialized = True
                logger.info("SHAP explainer initialized with transformer pipeline")
                return True
        except (ImportError, Exception) as e:
            logger.warning(f"SHAP not available ({e}), using keyword importance fallback")

        self._initialized = False
        return False

    def explain(
        self,
        clause_text: str,
        clause_id: str,
        clause_type: str,
        classification_result=None,
    ) -> ExplanationResult:
        """
        Generate SHAP explanation for a clause.

        Args:
            clause_text: The clause text.
            clause_id: Clause identifier.
            clause_type: Predicted clause type.
            classification_result: Optional ClassificationResult for context.

        Returns:
            ExplanationResult with token-level importance scores.
        """
        if self._initialized and self._shap_explainer is not None:
            return self._explain_with_shap(clause_text, clause_id, clause_type)
        else:
            return self._explain_with_keywords(clause_text, clause_id, clause_type)

    def _explain_with_shap(
        self, text: str, clause_id: str, clause_type: str
    ) -> ExplanationResult:
        """SHAP-based explanation using transformer pipeline."""
        try:
            import shap
            import numpy as np

            # Truncate for SHAP (computationally expensive)
            truncated = text[:500]

            shap_values = self._shap_explainer([truncated])

            # Extract token importances for the predicted class
            tokens = shap_values.data[0]
            values = shap_values.values[0]

            # If multi-class, take the predicted class dimension
            if len(values.shape) > 1:
                values = values[:, 0]  # First class

            token_importances = []
            for token, val in zip(tokens, values):
                if token in ("[CLS]", "[SEP]", "[PAD]"):
                    continue
                token_importances.append(TokenImportance(
                    token=token,
                    importance=float(val),
                    is_positive=float(val) > 0,
                ))

            top_pos = sorted(
                [t for t in token_importances if t.is_positive],
                key=lambda x: x.importance,
                reverse=True,
            )[:10]

            top_neg = sorted(
                [t for t in token_importances if not t.is_positive],
                key=lambda x: x.importance,
            )[:5]

            html = self._render_highlighted_text(text, token_importances)

            return ExplanationResult(
                clause_id=clause_id,
                clause_type=clause_type,
                token_importances=token_importances,
                top_positive_tokens=top_pos,
                top_negative_tokens=top_neg,
                explanation_html=html,
                method="shap_bert",
            )

        except Exception as e:
            logger.error(f"SHAP error: {e}")
            return self._explain_with_keywords(text, clause_id, clause_type)

    def _explain_with_keywords(
        self, text: str, clause_id: str, clause_type: str
    ) -> ExplanationResult:
        """
        Keyword-importance fallback. Highlights legal risk terms.

        Uses a curated set of high-risk terms with pre-assigned importance scores.
        """
        HIGH_RISK_TERMS = {
            # Critical terms (importance 0.9-1.0)
            "indemnif": 0.95,
            "unlimited liability": 0.95,
            "work for hire": 0.92,
            "assigns all right title": 0.90,
            "irrevocable": 0.88,
            "worldwide": 0.85,

            # High risk (0.7-0.89)
            "automatically renew": 0.82,
            "auto-renew": 0.82,
            "sole discretion": 0.80,
            "shall not exceed": 0.78,
            "non-compete": 0.78,
            "noncompete": 0.78,
            "anti-assignment": 0.75,
            "without consent": 0.72,
            "perpetual": 0.70,

            # Medium risk (0.4-0.69)
            "governing law": 0.60,
            "arbitration": 0.58,
            "confidential": 0.50,
            "intellectual property": 0.55,
            "termination": 0.45,
            "exclusiv": 0.60,
            "notice period": 0.55,
            "days prior": 0.52,
            "liquidated damages": 0.65,
            "change of control": 0.62,

            # Protective terms (negative importance = reduces risk)
            "mutual": -0.40,
            "reasonable": -0.35,
            "gross negligence": -0.45,  # Carve-out = good
            "willful misconduct": -0.45,
            "not be unreasonably": -0.38,
        }

        text_lower = text.lower()
        words = text.split()
        token_importances = []

        for word in words:
            word_clean = word.lower().strip(".,;:()[]\"'")
            importance = 0.0

            # Check if word is part of a high-risk term
            for term, score in HIGH_RISK_TERMS.items():
                if term in word_clean or word_clean in term:
                    importance = max(importance, score, key=abs) if importance != 0 else score

            token_importances.append(TokenImportance(
                token=word,
                importance=importance,
                is_positive=importance > 0,
            ))

        # Multi-word phrase scanning
        text_lower_words = text_lower.split()
        phrase_highlights = {}
        for term, score in HIGH_RISK_TERMS.items():
            term_words = term.split()
            if len(term_words) > 1:
                for i in range(len(text_lower_words) - len(term_words) + 1):
                    chunk = " ".join(text_lower_words[i:i + len(term_words)])
                    if term in chunk:
                        for j in range(len(term_words)):
                            phrase_highlights[i + j] = score

        # Apply phrase highlights
        for idx, score in phrase_highlights.items():
            if idx < len(token_importances):
                token_importances[idx].importance = score
                token_importances[idx].is_positive = score > 0

        top_pos = sorted(
            [t for t in token_importances if t.importance > 0.3],
            key=lambda x: x.importance,
            reverse=True,
        )[:10]

        top_neg = sorted(
            [t for t in token_importances if t.importance < -0.2],
            key=lambda x: x.importance,
        )[:5]

        html = self._render_highlighted_text(text, token_importances)

        return ExplanationResult(
            clause_id=clause_id,
            clause_type=clause_type,
            token_importances=token_importances,
            top_positive_tokens=top_pos,
            top_negative_tokens=top_neg,
            explanation_html=html,
            method="keyword_importance",
        )

    def _render_highlighted_text(
        self, text: str, token_importances: list[TokenImportance]
    ) -> str:
        """
        Render text with HTML color highlights based on SHAP/keyword importance.

        Red = increases risk, green = decreases risk.
        Opacity proportional to importance magnitude.
        """
        words = text.split()
        if len(words) != len(token_importances):
            # Fallback: just return plain text
            return f"<p>{text}</p>"

        html_parts = []
        for word, ti in zip(words, token_importances):
            abs_imp = abs(ti.importance)

            if abs_imp < 0.1:
                html_parts.append(word)
            elif ti.is_positive:
                # Red highlight for risk-increasing terms
                opacity = min(0.85, abs_imp)
                alpha = int(opacity * 255)
                html_parts.append(
                    f'<mark style="background-color: rgba(239,68,68,{opacity:.2f}); '
                    f'padding: 1px 3px; border-radius: 3px; font-weight: 600;" '
                    f'title="Risk factor: {ti.importance:.2f}">{word}</mark>'
                )
            else:
                # Green highlight for risk-reducing terms
                opacity = min(0.85, abs_imp)
                html_parts.append(
                    f'<mark style="background-color: rgba(34,197,94,{opacity:.2f}); '
                    f'padding: 1px 3px; border-radius: 3px;" '
                    f'title="Risk reducer: {ti.importance:.2f}">{word}</mark>'
                )

        return "<p style='line-height: 1.8;'>" + " ".join(html_parts) + "</p>"
