"""
Tests for Contract Risk Analyzer core modules.

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=src --cov-report=term-missing
"""

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_CONTRACT = """
SERVICE AGREEMENT

This Agreement is entered into between Acme Corp and GlobalCo Inc.

1. TERM AND RENEWAL
This Agreement shall automatically renew for successive one-year terms unless
either party provides written notice of non-renewal at least 15 days prior to
the end of the then-current term.

2. INDEMNIFICATION
Service Provider shall indemnify, defend, and hold harmless Client from and
against any claims arising out of performance under this Agreement. Such
indemnification shall be irrevocable.

3. LIMITATION OF LIABILITY
The total aggregate liability shall not exceed the fees paid in the one month
preceding the claim.

4. NON-COMPETE
During the term and for three (3) years following termination, Service Provider
shall not engage in any competing business on a worldwide basis.

5. GOVERNING LAW
This Agreement shall be governed by the laws of the State of Delaware.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Clause Segmenter Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestClauseSegmenter:
    def setup_method(self):
        from extraction.clause_segmenter import ClauseSegmenter
        self.segmenter = ClauseSegmenter(use_spacy=False)

    def test_segments_numbered_clauses(self):
        result = self.segmenter.segment(SAMPLE_CONTRACT)
        assert result.total_clauses >= 3, "Should detect at least 3 numbered clauses"

    def test_clause_has_required_fields(self):
        result = self.segmenter.segment(SAMPLE_CONTRACT)
        clause = result.clauses[0]
        assert hasattr(clause, "id")
        assert hasattr(clause, "raw_text")
        assert hasattr(clause, "word_count")
        assert clause.word_count > 0

    def test_method_detection(self):
        result = self.segmenter.segment(SAMPLE_CONTRACT)
        assert result.method_used in ("numbered", "paragraph", "hybrid")

    def test_empty_text(self):
        result = self.segmenter.segment("")
        assert result.total_clauses == 0
        assert result.method_used == "none"

    def test_min_word_filter(self):
        """Clauses below MIN_CLAUSE_WORDS should be filtered out."""
        result = self.segmenter.segment("1. Hi\n\n2. This is a much longer clause with many words that should pass the filter.")
        for clause in result.clauses:
            assert clause.word_count >= self.segmenter.MIN_CLAUSE_WORDS


# ─────────────────────────────────────────────────────────────────────────────
# CUAD Classifier Tests (keyword fallback — no GPU required)
# ─────────────────────────────────────────────────────────────────────────────
class TestCUADClassifier:
    def setup_method(self):
        from classification.cuad_model import CUADModel
        self.model = CUADModel()
        # Deliberately don't call load() so keyword fallback is used

    def test_classifies_indemnification(self):
        text = "Service Provider shall indemnify, defend, and hold harmless Client from all claims."
        result = self.model.classify_clause(text, "test_001")
        assert result.predicted_type == "indemnification"
        assert result.confidence > 0

    def test_classifies_auto_renewal(self):
        text = "This Agreement shall automatically renew for successive one-year terms."
        result = self.model.classify_clause(text, "test_002")
        assert result.predicted_type == "renewal_term"

    def test_classifies_governing_law(self):
        text = "This Agreement shall be governed by the laws of the State of Delaware."
        result = self.model.classify_clause(text, "test_003")
        assert result.predicted_type in ("governing_law",)

    def test_classifies_non_compete(self):
        text = "Service Provider shall not engage in any competing or competitive activity for 3 years."
        result = self.model.classify_clause(text, "test_004")
        assert result.predicted_type == "non_compete"

    def test_returns_required_fields(self):
        text = "Any disputes shall be resolved by binding arbitration."
        result = self.model.classify_clause(text, "test_005")
        assert hasattr(result, "predicted_type")
        assert hasattr(result, "confidence")
        assert hasattr(result, "top_predictions")
        assert 0.0 <= result.confidence <= 1.0

    def test_empty_clause(self):
        result = self.model.classify_clause("", "test_empty")
        assert result.predicted_type == "unknown"

    def test_batch_classify(self):
        from extraction.clause_segmenter import ClauseSegmenter
        segmenter = ClauseSegmenter(use_spacy=False)
        seg_result = segmenter.segment(SAMPLE_CONTRACT)
        results = self.model.batch_classify(seg_result.clauses, show_progress=False)
        assert len(results) == seg_result.total_clauses


# ─────────────────────────────────────────────────────────────────────────────
# Risk Scorer Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestRiskScorer:
    def setup_method(self):
        from risk.risk_scorer import RiskScorer
        from classification.cuad_model import ClassificationResult
        self.scorer = RiskScorer()
        self.ClassificationResult = ClassificationResult

    def _make_result(self, clause_type, text, confidence=0.8):
        from classification.cuad_model import ClassificationResult
        return ClassificationResult(
            clause_id="test",
            clause_text=text,
            predicted_type=clause_type,
            predicted_label=clause_type,
            confidence=confidence,
            top_predictions=[],
            model_used="test",
        )

    def test_critical_types_score_high(self):
        result = self._make_result(
            "indemnification",
            "Service Provider shall indemnify and hold harmless Client from all claims irrevocably.",
        )
        cr = self.scorer.score_clause(result)
        assert cr.risk_score >= 60, f"Indemnification score should be high, got {cr.risk_score}"

    def test_low_types_score_low(self):
        result = self._make_result(
            "parties",
            "This Agreement is entered into between Acme Corp and GlobalCo Inc.",
        )
        cr = self.scorer.score_clause(result)
        assert cr.risk_score <= 40, f"Parties clause should be low risk, got {cr.risk_score}"

    def test_auto_renewal_flag_detected(self):
        result = self._make_result(
            "renewal_term",
            "This Agreement shall automatically renew unless 10 days written notice is given.",
        )
        cr = self.scorer.score_clause(result)
        flag_ids = {f.flag_id for f in cr.flags}
        assert "auto_renewal" in flag_ids
        assert "short_notice" in flag_ids or "medium_notice" in flag_ids

    def test_sole_discretion_flag(self):
        result = self._make_result(
            "anti_assignment",
            "Assignment requires prior written consent at Client's sole discretion.",
        )
        cr = self.scorer.score_clause(result)
        flag_ids = {f.flag_id for f in cr.flags}
        assert "sole_discretion" in flag_ids

    def test_score_bounds(self):
        for clause_type in ["indemnification", "non_compete", "parties", "governing_law"]:
            result = self._make_result(clause_type, "Test clause text for testing purposes only.")
            cr = self.scorer.score_clause(result)
            assert 0 <= cr.risk_score <= 100, f"Score out of bounds: {cr.risk_score}"

    def test_contract_level_scoring(self):
        texts = [
            ("indemnification", "Indemnify, defend, and hold harmless from all claims irrevocably worldwide."),
            ("renewal_term", "Automatically renews unless 10 days notice given."),
            ("parties", "Between Acme Corp and GlobalCo Inc."),
        ]
        results = [self._make_result(t, txt) for t, txt in texts]
        contract = self.scorer.score_contract(results)

        assert 0 <= contract.overall_score <= 100
        assert contract.overall_level in ("low", "medium", "high", "critical")
        assert len(contract.clause_risks) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Redline Diff Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestRedlineDiff:
    def setup_method(self):
        from risk.redline_diff import RedlineDiff
        self.differ = RedlineDiff()

    def test_identical_texts(self):
        text = "This is the same contract text."
        result = self.differ.diff(text, text)
        assert result.change_summary["similarity_ratio"] == 1.0
        assert result.change_summary["added_lines"] == 0

    def test_detects_changes(self):
        orig = "The liability cap is $10,000."
        rev = "The liability cap is $50,000."
        result = self.differ.diff(orig, rev)
        assert result.change_summary["similarity_ratio"] < 1.0

    def test_html_outputs_populated(self):
        orig = "Original contract text with several clauses."
        rev = "Revised contract text with updated clauses and new terms."
        result = self.differ.diff(orig, rev)
        assert len(result.inline_html) > 0
        assert len(result.side_by_side_html) > 0
        assert "<del" in result.inline_html or "<ins" in result.inline_html


# ─────────────────────────────────────────────────────────────────────────────
# SHAP Explainer Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestSHAPExplainer:
    def setup_method(self):
        from explainability.shap_explainer import SHAPExplainer
        self.explainer = SHAPExplainer()

    def test_keyword_explanation(self):
        text = "Service Provider shall indemnify and hold harmless Client from claims. Sole discretion of Client."
        result = self.explainer.explain(text, "test_001", "indemnification")
        assert len(result.token_importances) > 0
        assert len(result.explanation_html) > 0

    def test_finds_high_risk_terms(self):
        text = "The agreement shall automatically renew and indemnify all parties irrevocably."
        result = self.explainer.explain(text, "test_002", "renewal_term")
        high_imp_tokens = [t.token.lower() for t in result.top_positive_tokens]
        # At least one high-risk term should be flagged
        risk_terms = {"indemnify", "automatically", "renew", "irrevocably"}
        assert any(t in high_imp_tokens for t in risk_terms)

    def test_html_contains_highlights(self):
        text = "Indemnification clause: Service Provider shall indemnify Client for all claims."
        result = self.explainer.explain(text, "test_003", "indemnification")
        assert "<mark" in result.explanation_html


# ─────────────────────────────────────────────────────────────────────────────
# Integration Test
# ─────────────────────────────────────────────────────────────────────────────
class TestIntegration:
    """End-to-end pipeline test (no PDF, no GPU required)."""

    def test_full_pipeline_on_sample_contract(self):
        from extraction.clause_segmenter import ClauseSegmenter
        from classification.cuad_model import CUADModel
        from risk.risk_scorer import RiskScorer
        from explainability.shap_explainer import SHAPExplainer

        segmenter = ClauseSegmenter(use_spacy=False)
        classifier = CUADModel()
        scorer = RiskScorer()
        explainer = SHAPExplainer()

        # Segment
        seg = segmenter.segment(SAMPLE_CONTRACT)
        assert seg.total_clauses >= 3

        # Classify
        classifications = classifier.batch_classify(seg.clauses, show_progress=False)
        assert len(classifications) == seg.total_clauses

        # Score
        contract_risk = scorer.score_contract(classifications)
        assert 0 <= contract_risk.overall_score <= 100
        assert len(contract_risk.clause_risks) == seg.total_clauses

        # Explain
        for cr in contract_risk.clause_risks[:3]:
            exp = explainer.explain(cr.raw_text, cr.clause_id, cr.clause_type)
            assert len(exp.explanation_html) > 0

        # Verify high-risk clauses were detected
        # (sample contract has indemnification + auto-renewal)
        high_or_critical = [
            cr for cr in contract_risk.clause_risks
            if cr.risk_level in ("high", "critical")
        ]
        assert len(high_or_critical) >= 1, "Should detect at least 1 high-risk clause"
