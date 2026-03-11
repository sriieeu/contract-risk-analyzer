"""
CUAD Model — wraps HuggingFace BERT fine-tuned on the Contract Understanding
Atticus Dataset (CUAD) for clause-type classification.

Two modes:
  1. Use a pre-trained CUAD checkpoint (recommended for production)
  2. Fine-tune from scratch on the CUAD dataset

Reference: https://huggingface.co/datasets/cuad
Paper: https://arxiv.org/abs/2103.06268
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import logging; logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pre-trained CUAD model on HuggingFace Hub
# ---------------------------------------------------------------------------
CUAD_HF_MODEL = "theatticusproject/cuad-qa"  # Official CUAD QA model
BERT_BASE = "bert-base-uncased"

# Map CUAD question categories to our label taxonomy
CUAD_QUESTION_MAP = {
    "Parties": "parties",
    "Agreement Date": "agreement_date",
    "Effective Date": "effective_date",
    "Expiration Date": "expiration_date",
    "Renewal Term": "renewal_term",
    "Notice Period to Terminate Renewal": "notice_period_to_terminate_renewal",
    "Governing Law": "governing_law",
    "Dispute Resolution": "dispute_resolution",
    "Anti-Assignment": "anti_assignment",
    "IP Ownership Assignment": "ip_ownership_assignment",
    "License Grant": "license_grant",
    "Non-Compete": "non_compete",
    "Exclusivity": "exclusivity",
    "No-Solicit Of Customers": "no_solicitation",
    "Non-Disparagement": "non_disparagement",
    "Limitation Of Liability": "limitation_of_liability",
    "Liquidated Damages": "liquidated_damages",
    "Warranty Duration": "warranty_duration",
    "Insurance": "insurance",
    "Audit Rights": "audit_rights",
    "Most Favored Nation": "most_favored_nation",
    "Cap On Liability": "cap_on_liability",
    "Indemnification": "indemnification",
    "Change Of Control": "change_of_control",
    "Termination For Convenience": "termination_for_convenience",
    "ROFR/ROFO/ROFN": "rofr_rofo_rofn",
    "Minimum Commitment": "minimum_commitment",
    "Revenue/Profit Sharing": "revenue_profit_sharing",
    "Price Restrictions": "price_restrictions",
    "Confidentiality": "confidentiality",
}


@dataclass
class ClassificationResult:
    """Result from classifying a single clause."""
    clause_id: str
    clause_text: str
    predicted_type: str          # e.g. "indemnification"
    predicted_label: str         # e.g. "Indemnification"
    confidence: float            # 0.0 - 1.0
    top_predictions: list[dict]  # Top-k predictions with scores
    model_used: str              # "cuad_bert" | "keyword_fallback"


class CUADModel:
    """
    CUAD-based clause type classifier.

    Uses the official CUAD QA model from HuggingFace to identify which
    clause types are present in a given text span.

    For production use, fine-tune bert-base-uncased on the CUAD dataset
    using train_cuad.py and point model_path to your checkpoint.

    Usage:
        model = CUADModel()
        model.load()
        result = model.classify_clause("This Agreement shall automatically renew...")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_gpu: bool = False,
        max_length: int = 512,
    ):
        self.model_path = model_path or CUAD_HF_MODEL
        self.use_gpu = use_gpu
        self.max_length = max_length
        self._pipeline = None
        self._tokenizer = None
        self._model = None
        self._loaded = False

        # Load label taxonomy
        labels_path = Path(__file__).parent.parent.parent / "data" / "cuad_labels.json"
        with open(labels_path) as f:
            self._taxonomy = json.load(f)

    def load(self) -> None:
        """Load the model. Call this once before classifying."""
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
            import torch

            device = 0 if self.use_gpu and torch.cuda.is_available() else -1
            device_name = "GPU" if device == 0 else "CPU"

            logger.info(f"Loading CUAD model from '{self.model_path}' on {device_name}...")

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForQuestionAnswering.from_pretrained(self.model_path)

            self._pipeline = pipeline(
                "question-answering",
                model=self._model,
                tokenizer=self._tokenizer,
                device=device,
            )
            self._loaded = True
            logger.info(f"CUAD model loaded successfully")

        except Exception as e:
            logger.warning(
                f"Could not load transformer model: {e}\n"
                "Falling back to keyword-based classifier."
            )
            self._loaded = False

    def classify_clause(self, clause_text: str, clause_id: str = "unknown") -> ClassificationResult:
        """
        Classify a clause using CUAD QA-style inference.

        The CUAD model is a QA model: each clause type is a "question"
        and we ask the model if the clause contains that type.
        The clause with the highest answer score is the predicted type.

        Args:
            clause_text: The text of a single contract clause.
            clause_id: Identifier for the clause.

        Returns:
            ClassificationResult with predicted type and confidence.
        """
        if not clause_text.strip():
            return self._empty_result(clause_id, clause_text)

        if self._loaded and self._pipeline is not None:
            return self._classify_with_bert(clause_text, clause_id)
        else:
            return self._classify_with_keywords(clause_text, clause_id)

    def _classify_with_bert(self, text: str, clause_id: str) -> ClassificationResult:
        """
        CUAD QA-style: ask each clause-type question and rank by score.
        """
        # CUAD questions for each clause type
        cuad_questions = {
            "renewal_term": "Is there a renewal term or automatic renewal clause?",
            "indemnification": "Does this clause contain an indemnification obligation?",
            "limitation_of_liability": "Is there a limitation of liability or liability cap?",
            "cap_on_liability": "What is the cap on liability?",
            "non_compete": "Is there a non-compete restriction?",
            "anti_assignment": "Is there an anti-assignment restriction?",
            "ip_ownership_assignment": "Is there an IP ownership assignment?",
            "governing_law": "What is the governing law or jurisdiction?",
            "dispute_resolution": "How are disputes resolved?",
            "exclusivity": "Is there an exclusivity obligation?",
            "termination_for_convenience": "Is there a termination for convenience right?",
            "change_of_control": "Is there a change of control provision?",
            "confidentiality": "Is there a confidentiality or NDA obligation?",
            "data_breach_notification": "Is there a data breach notification requirement?",
            "audit_rights": "Are there audit rights?",
            "minimum_commitment": "Is there a minimum purchase or payment commitment?",
            "insurance": "Are there insurance requirements?",
            "most_favored_nation": "Is there a most favored nation clause?",
            "liquidated_damages": "Are there liquidated damages?",
            "warranty_duration": "What is the warranty duration?",
        }

        scores = {}
        try:
            # Truncate text for QA model
            truncated = text[:2000]  # BERT max context

            for clause_type, question in cuad_questions.items():
                try:
                    result = self._pipeline(
                        question=question,
                        context=truncated,
                        max_answer_len=50,
                    )
                    # Score is the model's confidence it found the answer
                    scores[clause_type] = result["score"]
                except Exception:
                    scores[clause_type] = 0.0

        except Exception as e:
            logger.error(f"BERT inference error: {e}")
            return self._classify_with_keywords(text, clause_id)

        # Sort by score
        sorted_preds = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_type, best_score = sorted_preds[0]

        # If confidence is very low, use keyword fallback
        if best_score < 0.05:
            return self._classify_with_keywords(text, clause_id)

        top_preds = [
            {
                "type": t,
                "label": self._taxonomy["clause_types"].get(t, {}).get("label", t),
                "score": float(s),
            }
            for t, s in sorted_preds[:5]
        ]

        label = self._taxonomy["clause_types"].get(best_type, {}).get("label", best_type)

        return ClassificationResult(
            clause_id=clause_id,
            clause_text=text,
            predicted_type=best_type,
            predicted_label=label,
            confidence=float(best_score),
            top_predictions=top_preds,
            model_used="cuad_bert",
        )

    def _classify_with_keywords(self, text: str, clause_id: str) -> ClassificationResult:
        """
        Keyword-based fallback classifier.
        Maps clause types to characteristic legal terms.
        """
        text_lower = text.lower()

        keyword_map = {
            "indemnification": [
                "indemnif", "indemnification", "hold harmless", "defend and indemnify",
                "indemnitor", "indemnitee",
            ],
            "limitation_of_liability": [
                "limitation of liability", "limit of liability", "not be liable",
                "shall not be liable", "excludes liability",
            ],
            "cap_on_liability": [
                "aggregate liability", "total liability", "maximum liability",
                "liability shall not exceed", "liability is capped",
            ],
            "non_compete": [
                "non-compete", "noncompete", "not compete", "competing business",
                "competitive activity", "restricted business",
            ],
            "renewal_term": [
                "automatically renew", "auto-renew", "renewal term", "successive terms",
                "renews automatically", "evergreen",
            ],
            "anti_assignment": [
                "may not assign", "shall not assign", "anti-assignment",
                "without prior written consent", "assignment is prohibited",
            ],
            "ip_ownership_assignment": [
                "intellectual property", "all right title and interest", "work for hire",
                "assigns to", "ip ownership", "assignment of rights",
            ],
            "governing_law": [
                "governed by the laws", "governing law", "laws of the state",
                "subject to the laws", "jurisdiction of",
            ],
            "dispute_resolution": [
                "arbitration", "mediation", "dispute resolution", "binding arbitration",
                "aaa rules", "jams", "resolve disputes",
            ],
            "exclusivity": [
                "exclusive", "exclusively", "sole and exclusive", "exclusivity period",
                "exclusive right",
            ],
            "termination_for_convenience": [
                "terminate for convenience", "termination for convenience",
                "terminate without cause", "terminate at will",
            ],
            "change_of_control": [
                "change of control", "change in control", "merger", "acquisition",
                "majority of shares", "controlling interest",
            ],
            "confidentiality": [
                "confidential", "confidentiality", "non-disclosure", "nda",
                "proprietary information", "trade secret",
            ],
            "data_breach_notification": [
                "data breach", "security breach", "notify within", "breach notification",
                "unauthorized access",
            ],
            "audit_rights": [
                "audit right", "right to audit", "inspect records", "examine books",
                "accounting records",
            ],
            "minimum_commitment": [
                "minimum purchase", "minimum order", "minimum commitment",
                "purchase obligation", "minimum revenue",
            ],
            "insurance": [
                "insurance", "general liability", "errors and omissions",
                "workers compensation", "certificate of insurance",
            ],
            "notice_period_to_terminate_renewal": [
                "notice of non-renewal", "written notice", "days prior to",
                "notice period", "terminate notice",
            ],
            "liquidated_damages": [
                "liquidated damages", "pre-agreed damages", "agreed damages",
                "as liquidated",
            ],
            "parties": [
                "this agreement is entered", "between", "hereinafter referred to",
                "the parties agree",
            ],
            "governing_law": [
                "governed by", "governing law", "laws of",
            ],
        }

        scores = {}
        for clause_type, keywords in keyword_map.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                # Normalize: more keywords = higher confidence (max out at 5)
                scores[clause_type] = min(score / 5.0, 0.9)

        if not scores:
            # Generic clause
            return ClassificationResult(
                clause_id=clause_id,
                clause_text=text,
                predicted_type="parties",
                predicted_label="General Provision",
                confidence=0.1,
                top_predictions=[],
                model_used="keyword_fallback",
            )

        sorted_preds = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_type, best_score = sorted_preds[0]

        top_preds = [
            {
                "type": t,
                "label": self._taxonomy["clause_types"].get(t, {}).get("label", t),
                "score": float(s),
            }
            for t, s in sorted_preds[:5]
        ]

        label = self._taxonomy["clause_types"].get(best_type, {}).get("label", best_type)

        return ClassificationResult(
            clause_id=clause_id,
            clause_text=text,
            predicted_type=best_type,
            predicted_label=label,
            confidence=float(best_score),
            top_predictions=top_preds,
            model_used="keyword_fallback",
        )

    def _empty_result(self, clause_id: str, text: str) -> ClassificationResult:
        return ClassificationResult(
            clause_id=clause_id,
            clause_text=text,
            predicted_type="unknown",
            predicted_label="Unknown",
            confidence=0.0,
            top_predictions=[],
            model_used="none",
        )

    def batch_classify(self, clauses: list, show_progress: bool = True) -> list[ClassificationResult]:
        """
        Classify a list of Clause objects in batch.

        Args:
            clauses: List of Clause objects from ClauseSegmenter.
            show_progress: Show tqdm progress bar.

        Returns:
            List of ClassificationResult objects.
        """
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(x, **kw): return x

        results = []
        iterator = tqdm(clauses, desc="Classifying clauses") if show_progress else clauses

        for clause in iterator:
            result = self.classify_clause(clause.raw_text, clause.id)
            results.append(result)

        return results
