"""
Risk Scoring Engine — converts clause classifications into risk scores
with plain-English explanations.

Risk score (0-100):
  0-25:   Low    — standard clause, minimal concern
  26-55:  Medium — review advised, potential gotchas
  56-80:  High   — significant risk, negotiate or reject
  81-100: Critical — contract-blocking risk, must address

Score is composed of:
  - Base risk from clause type (from CUAD taxonomy)
  - Severity modifiers from specific language patterns
  - Contextual amplifiers (e.g. one-sided vs. mutual)
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import logging; logger = logging.getLogger(__name__)


@dataclass
class RiskFlag:
    """A specific risk indicator found in clause text."""
    flag_id: str
    severity: str           # "low" | "medium" | "high" | "critical"
    title: str
    explanation: str
    matched_text: Optional[str] = None
    score_impact: int = 0


@dataclass
class ClauseRisk:
    """Risk assessment for a single clause."""
    clause_id: str
    clause_type: str
    clause_label: str
    raw_text: str
    risk_score: int              # 0-100
    risk_level: str              # "low" | "medium" | "high" | "critical"
    risk_color: str              # hex color
    base_score: int              # Score from clause type alone
    modifier_score: int          # Score from language analysis
    flags: list[RiskFlag]        # Specific risk flags
    plain_english: str           # Human-readable summary
    recommendations: list[str]   # What to do about this


@dataclass
class ContractRisk:
    """Overall risk assessment for an entire contract."""
    overall_score: int
    overall_level: str
    clause_risks: list[ClauseRisk]
    critical_clauses: list[ClauseRisk]
    high_risk_clauses: list[ClauseRisk]
    summary: str
    top_concerns: list[str]
    clause_type_distribution: dict[str, int]


class RiskScorer:
    """
    Scores contract clauses for risk using rule-based + pattern analysis.

    Usage:
        scorer = RiskScorer()
        contract_risk = scorer.score_contract(classification_results)
    """

    def __init__(self):
        labels_path = Path(__file__).parent.parent.parent / "data" / "cuad_labels.json"
        with open(labels_path) as f:
            self._taxonomy = json.load(f)

        self._base_scores = {
            "low": 15,
            "medium": 40,
            "high": 65,
            "critical": 85,
        }

        self._risk_colors = {
            "low": "#22c55e",
            "medium": "#f59e0b",
            "high": "#ef4444",
            "critical": "#7f1d1d",
        }

    def score_clause(self, classification_result) -> ClauseRisk:
        """Score a single classified clause."""
        clause_type = classification_result.predicted_type
        clause_text = classification_result.clause_text
        clause_id = classification_result.clause_id

        # Get base risk level from taxonomy
        type_meta = self._taxonomy["clause_types"].get(clause_type, {})
        base_level = type_meta.get("risk_level", "low")
        base_score = self._base_scores[base_level]
        base_explanation = type_meta.get("risk_explanation", "")
        label = type_meta.get("label", clause_type)

        # Analyze text for modifiers
        flags = self._detect_risk_flags(clause_type, clause_text)
        modifier_score = sum(f.score_impact for f in flags)

        # Clamp final score to 0-100
        final_score = max(0, min(100, base_score + modifier_score))

        # Adjust confidence: low confidence = regress toward base score
        confidence_weight = max(0.3, classification_result.confidence)
        adjusted_score = int(base_score * (1 - confidence_weight) + final_score * confidence_weight)

        risk_level = self._score_to_level(adjusted_score)
        risk_color = self._risk_colors[risk_level]

        plain_english = self._generate_plain_english(
            clause_type, clause_text, flags, base_explanation, risk_level
        )

        recommendations = self._generate_recommendations(clause_type, flags, risk_level)

        return ClauseRisk(
            clause_id=clause_id,
            clause_type=clause_type,
            clause_label=label,
            raw_text=clause_text,
            risk_score=adjusted_score,
            risk_level=risk_level,
            risk_color=risk_color,
            base_score=base_score,
            modifier_score=modifier_score,
            flags=flags,
            plain_english=plain_english,
            recommendations=recommendations,
        )

    def score_contract(self, classification_results: list) -> ContractRisk:
        """Score all clauses and produce a contract-level risk assessment."""
        clause_risks = [self.score_clause(r) for r in classification_results]

        if not clause_risks:
            return ContractRisk(
                overall_score=0,
                overall_level="low",
                clause_risks=[],
                critical_clauses=[],
                high_risk_clauses=[],
                summary="No clauses were analyzed.",
                top_concerns=[],
                clause_type_distribution={},
            )

        # Overall score: weighted average (critical/high clauses count more)
        weight_map = {"low": 0.5, "medium": 1.0, "high": 2.0, "critical": 3.0}
        weighted_sum = sum(
            cr.risk_score * weight_map[cr.risk_level] for cr in clause_risks
        )
        total_weight = sum(weight_map[cr.risk_level] for cr in clause_risks)
        overall_score = int(weighted_sum / total_weight) if total_weight > 0 else 0

        # Also factor in: does the contract have critical clauses?
        critical = [cr for cr in clause_risks if cr.risk_level == "critical"]
        high_risk = [cr for cr in clause_risks if cr.risk_level == "high"]

        if critical:
            overall_score = max(overall_score, 70)  # Can't be low if critical clauses exist
        if len(high_risk) >= 3:
            overall_score = max(overall_score, 55)

        overall_level = self._score_to_level(overall_score)

        # Clause type distribution
        type_dist = {}
        for cr in clause_risks:
            type_dist[cr.clause_label] = type_dist.get(cr.clause_label, 0) + 1

        # Top concerns
        top_concerns = []
        for cr in sorted(clause_risks, key=lambda x: x.risk_score, reverse=True)[:5]:
            if cr.risk_score >= 40:
                top_concerns.append(f"{cr.clause_label}: {cr.plain_english[:100]}...")

        summary = self._generate_contract_summary(overall_score, overall_level, critical, high_risk, clause_risks)

        return ContractRisk(
            overall_score=overall_score,
            overall_level=overall_level,
            clause_risks=clause_risks,
            critical_clauses=critical,
            high_risk_clauses=high_risk,
            summary=summary,
            top_concerns=top_concerns,
            clause_type_distribution=type_dist,
        )

    def _detect_risk_flags(self, clause_type: str, text: str) -> list[RiskFlag]:
        """Detect specific risk indicators in clause text."""
        flags = []
        text_lower = text.lower()

        # -----------------------------------------------------------------------
        # Universal flags (apply to any clause type)
        # -----------------------------------------------------------------------
        if re.search(r"\bsole\s+discretion\b", text_lower):
            flags.append(RiskFlag(
                flag_id="sole_discretion",
                severity="high",
                title="Sole Discretion Language",
                explanation="'Sole discretion' gives one party unilateral authority with no obligation to act reasonably.",
                matched_text=self._find_context(text_lower, "sole discretion"),
                score_impact=15,
            ))

        if re.search(r"\bperpetual\b", text_lower):
            flags.append(RiskFlag(
                flag_id="perpetual",
                severity="medium",
                title="Perpetual Obligation",
                explanation="Perpetual terms create indefinite obligations with no expiry.",
                score_impact=10,
            ))

        if re.search(r"\birrevocable\b", text_lower):
            flags.append(RiskFlag(
                flag_id="irrevocable",
                severity="high",
                title="Irrevocable Provision",
                explanation="Irrevocable rights cannot be undone even for cause.",
                score_impact=15,
            ))

        if re.search(r"\bunlimited\s+liabilit", text_lower):
            flags.append(RiskFlag(
                flag_id="unlimited_liability",
                severity="critical",
                title="Unlimited Liability Exposure",
                explanation="No cap on liability creates unconstrained financial exposure.",
                score_impact=30,
            ))

        # -----------------------------------------------------------------------
        # Clause-specific flags
        # -----------------------------------------------------------------------
        if clause_type in ("renewal_term", "notice_period_to_terminate_renewal"):
            # Auto-renewal
            if re.search(r"automatically\s+renew|auto.?renew|evergreen", text_lower):
                flags.append(RiskFlag(
                    flag_id="auto_renewal",
                    severity="high",
                    title="Automatic Renewal",
                    explanation="Contract renews automatically without action. Miss the opt-out window and you're locked in.",
                    score_impact=20,
                ))

            # Very short notice window
            short_notice = re.search(
                r"(\d+)\s*(?:calendar\s+)?days?\s*(?:prior|before|advance)?\s*(?:written\s+)?notice",
                text_lower,
            )
            if short_notice:
                days = int(short_notice.group(1))
                if days < 30:
                    flags.append(RiskFlag(
                        flag_id="short_notice",
                        severity="critical",
                        title=f"Very Short Notice Window ({days} days)",
                        explanation=f"Only {days} days to cancel renewal. Easy to miss; automatic multi-year commitment triggered.",
                        matched_text=short_notice.group(0),
                        score_impact=25,
                    ))
                elif days < 60:
                    flags.append(RiskFlag(
                        flag_id="medium_notice",
                        severity="medium",
                        title=f"Short Notice Window ({days} days)",
                        explanation=f"{days} days to cancel renewal. Calendar this immediately upon signing.",
                        score_impact=10,
                    ))

        if clause_type in ("indemnification", "cap_on_liability", "limitation_of_liability"):
            # One-sided indemnification
            if not re.search(r"\bmutual\b|\beach\s+party\b|\bboth\s+parties\b", text_lower):
                if re.search(r"\bindemnif", text_lower):
                    flags.append(RiskFlag(
                        flag_id="one_sided_indemnification",
                        severity="high",
                        title="One-Sided Indemnification",
                        explanation="Indemnification appears to flow one way. Mutual indemnification is the market standard.",
                        score_impact=20,
                    ))

            # Gross negligence carve-out missing
            if re.search(r"\bindemnif", text_lower):
                if not re.search(r"gross\s+negligence|willful\s+misconduct|fraud", text_lower):
                    flags.append(RiskFlag(
                        flag_id="no_gross_neg_carveout",
                        severity="high",
                        title="No Gross Negligence Carve-Out",
                        explanation="Indemnification without carve-outs for gross negligence or willful misconduct creates extreme exposure.",
                        score_impact=15,
                    ))

        if clause_type == "non_compete":
            # Duration check
            years_match = re.search(r"(\d+)\s*year", text_lower)
            if years_match:
                years = int(years_match.group(1))
                if years >= 3:
                    flags.append(RiskFlag(
                        flag_id="long_noncompete",
                        severity="critical",
                        title=f"Extended Non-Compete Duration ({years} years)",
                        explanation=f"A {years}-year non-compete is unusually long and may be unenforceable in some jurisdictions.",
                        score_impact=25,
                    ))

            # Geographic scope
            if re.search(r"\bworldwide\b|\bglobal\b|\buniversal\b", text_lower):
                flags.append(RiskFlag(
                    flag_id="global_noncompete",
                    severity="critical",
                    title="Global Non-Compete Scope",
                    explanation="Worldwide non-compete scope severely limits business activities across all markets.",
                    score_impact=25,
                ))

        if clause_type == "governing_law":
            # Foreign jurisdiction
            if re.search(r"\bdelaware\b", text_lower):
                flags.append(RiskFlag(
                    flag_id="delaware_law",
                    severity="low",
                    title="Delaware Governing Law",
                    explanation="Delaware is a common, well-understood commercial law jurisdiction. Generally favorable.",
                    score_impact=-5,
                ))
            elif re.search(r"\bforeign\b|\binternational\b", text_lower):
                flags.append(RiskFlag(
                    flag_id="foreign_jurisdiction",
                    severity="medium",
                    title="Potential Foreign Jurisdiction",
                    explanation="Foreign jurisdiction may require international legal counsel and complicate enforcement.",
                    score_impact=15,
                ))

        if clause_type == "ip_ownership_assignment":
            if re.search(r"\bwork\s+for\s+hire\b|work-for-hire", text_lower):
                flags.append(RiskFlag(
                    flag_id="work_for_hire",
                    severity="critical",
                    title="Work-for-Hire Designation",
                    explanation="Work-for-hire transfers full IP ownership to the commissioning party automatically. All created IP is theirs.",
                    score_impact=25,
                ))

            if re.search(r"\bassigns?\s+all|all\s+right[,s]?\s+title\s+and\s+interest", text_lower):
                flags.append(RiskFlag(
                    flag_id="all_ip_assignment",
                    severity="high",
                    title="Full IP Assignment",
                    explanation="Assignment of 'all right, title and interest' transfers complete ownership of all IP created.",
                    score_impact=20,
                ))

        return flags

    def _score_to_level(self, score: int) -> str:
        if score <= 25:
            return "low"
        elif score <= 55:
            return "medium"
        elif score <= 80:
            return "high"
        else:
            return "critical"

    def _find_context(self, text: str, phrase: str, window: int = 50) -> str:
        """Find phrase in text with surrounding context."""
        idx = text.find(phrase)
        if idx == -1:
            return phrase
        start = max(0, idx - window)
        end = min(len(text), idx + len(phrase) + window)
        return "..." + text[start:end] + "..."

    def _generate_plain_english(
        self, clause_type: str, text: str, flags: list[RiskFlag],
        base_explanation: str, risk_level: str
    ) -> str:
        """Generate a human-readable explanation of the clause risk."""
        if not flags:
            return base_explanation or "Standard clause. Review to ensure it matches your expectations."

        flag_summaries = [f.explanation for f in flags[:3]]  # Top 3 flags
        combined = " ".join(flag_summaries)

        level_prefix = {
            "low": "✅ Low concern.",
            "medium": "⚠️ Review advised.",
            "high": "🔴 High risk.",
            "critical": "🚨 Critical risk — must address.",
        }[risk_level]

        return f"{level_prefix} {combined}"

    def _generate_recommendations(
        self, clause_type: str, flags: list[RiskFlag], risk_level: str
    ) -> list[str]:
        """Generate actionable recommendations."""
        recs = []

        flag_ids = {f.flag_id for f in flags}

        if "auto_renewal" in flag_ids:
            recs.append("Calendar the opt-out deadline immediately upon signing.")

        if "short_notice" in flag_ids or "medium_notice" in flag_ids:
            recs.append("Negotiate for a minimum 60-day notice period.")

        if "one_sided_indemnification" in flag_ids:
            recs.append("Request mutual indemnification or cap the indemnification obligation.")

        if "no_gross_neg_carveout" in flag_ids:
            recs.append("Add carve-outs for gross negligence, willful misconduct, and fraud.")

        if "unlimited_liability" in flag_ids:
            recs.append("Negotiate a liability cap — typically 12 months of fees paid.")

        if "work_for_hire" in flag_ids:
            recs.append("Negotiate a license-back provision to retain rights to use created IP.")

        if "all_ip_assignment" in flag_ids:
            recs.append("Limit IP assignment to deliverables specifically created for this engagement.")

        if "global_noncompete" in flag_ids:
            recs.append("Narrow non-compete scope to specific geographies where you compete.")

        if "long_noncompete" in flag_ids:
            recs.append("Negotiate non-compete duration to 12-18 months maximum.")

        if "sole_discretion" in flag_ids:
            recs.append("Replace 'sole discretion' with 'reasonable discretion not to be unreasonably withheld'.")

        if not recs and risk_level in ("high", "critical"):
            recs.append("Consult with legal counsel before accepting this clause.")

        if not recs:
            recs.append("Review clause language carefully to ensure it matches business expectations.")

        return recs

    def _generate_contract_summary(
        self, score: int, level: str, critical: list, high_risk: list, all_clauses: list
    ) -> str:
        """Generate a contract-level risk summary."""
        total = len(all_clauses)

        level_desc = {
            "low": "This contract presents minimal risk overall.",
            "medium": "This contract has moderate risk that warrants careful review.",
            "high": "This contract has significant risk areas that require attention before signing.",
            "critical": "This contract has critical risk issues that must be resolved before signing.",
        }[level]

        parts = [level_desc]

        if critical:
            clause_names = ", ".join(set(c.clause_label for c in critical[:3]))
            parts.append(f"Critical issues found in: {clause_names}.")

        if high_risk:
            clause_names = ", ".join(set(c.clause_label for c in high_risk[:4]))
            parts.append(f"High-risk clauses: {clause_names}.")

        parts.append(f"Analyzed {total} clauses. Overall risk score: {score}/100.")

        return " ".join(parts)
