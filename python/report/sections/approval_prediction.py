"""Approval prediction section — neural-network output from the planning oracle.

Displays the ML model's calibrated approval probability, confidence
interval, and council-level probability distribution (top councils ranked
by approval affinity).

When the neural network starts producing structured ``reasonings``, this
section will also render them as insights.  Until then the reasonings
subsection gracefully degrades to a placeholder note.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from report.models import (
    CouncilContext,
    Insight,
    Metric,
    OraclePrediction,
    SectionResult,
    format_reasonings,
)
from report.sections.base import BaseSection


class ApprovalPredictionSection(BaseSection):
    """Report section for planning-oracle neural-network predictions."""

    section_id = "approval_prediction"
    title = "Approval Prediction (Neural Network)"

    @property
    def required_tables(self) -> list[str]:
        return []

    def run(
        self,
        council: CouncilContext,
        data: dict[str, pd.DataFrame],
        *,
        oracle: Optional[OraclePrediction] = None,
    ) -> SectionResult:
        if oracle is None:
            return SectionResult(
                section_id=self.section_id,
                title=self.title,
                summary=(
                    "No neural-network prediction was provided for this report. "
                    "Run the planning-oracle inference pipeline and pass the "
                    "result to the report builder to populate this section."
                ),
                data_quality="unavailable",
                data_source="Planning Oracle neural network",
            )

        metrics = self._build_metrics(oracle)
        insights = self._build_insights(oracle, council)
        summary = self._build_summary(oracle, council)

        return SectionResult(
            section_id=self.section_id,
            title=self.title,
            summary=summary,
            metrics=metrics,
            insights=insights,
            data_quality="full",
            data_source="Planning Oracle neural network (calibrated)",
        )

    # ── Metrics ───────────────────────────────────────────────────────

    @staticmethod
    def _build_metrics(oracle: OraclePrediction) -> list[Metric]:
        prob = oracle.approval_probability
        ci_lo, ci_hi = oracle.confidence_interval

        direction = "positive" if prob >= 0.6 else ("negative" if prob < 0.4 else "neutral")

        metrics: list[Metric] = [
            Metric(
                label="Approval probability",
                value=round(prob * 100, 1),
                unit="%",
                context=f"95% CI: {ci_lo * 100:.1f}%–{ci_hi * 100:.1f}%",
                direction=direction,
            ),
            Metric(
                label="Confidence width",
                value=round((ci_hi - ci_lo) * 100, 1),
                unit="pp",
                context="Narrower is better",
                direction="positive" if (ci_hi - ci_lo) < 0.2 else "neutral",
            ),
        ]

        # Top council probability distribution
        if oracle.top_councils:
            best = oracle.top_councils[0]
            name = best.council_name or f"Council {best.council_id}"
            metrics.append(
                Metric(
                    label="Best council",
                    value=name,
                    context=f"Affinity score: {best.score:.2f}",
                    direction="positive",
                ),
            )

            if len(oracle.top_councils) >= 3:
                third = oracle.top_councils[2]
                spread = best.score - third.score
                metrics.append(
                    Metric(
                        label="Top-3 score spread",
                        value=round(spread, 3),
                        context="Difference between #1 and #3 affinity",
                        direction="neutral",
                    ),
                )

        return metrics

    # ── Insights ──────────────────────────────────────────────────────

    @staticmethod
    def _build_insights(
        oracle: OraclePrediction,
        council: CouncilContext,
    ) -> list[Insight]:
        insights: list[Insight] = []
        prob = oracle.approval_probability

        # Overall probability assessment
        if prob >= 0.75:
            insights.append(Insight(
                text=(
                    f"The neural network predicts a strong likelihood of approval "
                    f"({prob * 100:.0f}%). Historical patterns for comparable proposals "
                    f"in similar councils support this assessment."
                ),
                sentiment="positive",
            ))
        elif prob >= 0.5:
            insights.append(Insight(
                text=(
                    f"The model gives a moderate approval probability of {prob * 100:.0f}%. "
                    f"The outcome is not certain and may depend on site-specific factors "
                    f"not captured by the model."
                ),
                sentiment="neutral",
            ))
        else:
            insights.append(Insight(
                text=(
                    f"The neural network flags a below-average approval probability "
                    f"({prob * 100:.0f}%). Consider reviewing the proposal against local "
                    f"policy requirements before submission."
                ),
                sentiment="negative",
            ))

        # Council ranking insights
        if oracle.top_councils:
            top_names = [
                c.council_name or f"Council {c.council_id}"
                for c in oracle.top_councils[:5]
            ]
            insights.append(Insight(
                text=(
                    f"Top councils by approval affinity: "
                    f"{', '.join(top_names)}."
                ),
                sentiment="neutral",
            ))

            # Score distribution insight
            scores = [c.score for c in oracle.top_councils]
            if len(scores) >= 2:
                top_score = scores[0]
                median_score = scores[len(scores) // 2]
                if top_score - median_score > 0.15:
                    insights.append(Insight(
                        text=(
                            f"There is significant variation in council affinity scores "
                            f"(top: {top_score:.2f}, median: {median_score:.2f}). "
                            f"Council selection could materially affect approval chances."
                        ),
                        sentiment="neutral",
                    ))

        # Reasonings from the neural network (when available)
        if oracle.reasonings:
            reasoning_texts = format_reasonings(oracle.reasonings)
            for text in reasoning_texts:
                insights.append(Insight(text=text, sentiment="neutral"))
        else:
            insights.append(Insight(
                text=(
                    "Detailed model reasoning is not yet available. "
                    "Future model versions will provide factor-level explanations."
                ),
                sentiment="neutral",
            ))

        return insights

    # ── Summary ───────────────────────────────────────────────────────

    @staticmethod
    def _build_summary(
        oracle: OraclePrediction,
        council: CouncilContext,
    ) -> str:
        prob = oracle.approval_probability
        ci_lo, ci_hi = oracle.confidence_interval

        verdict = (
            "favourable" if prob >= 0.6
            else "uncertain" if prob >= 0.4
            else "challenging"
        )

        parts = [
            f"The planning-oracle neural network estimates a {prob * 100:.0f}% "
            f"probability of approval (95% CI: {ci_lo * 100:.0f}%–{ci_hi * 100:.0f}%), "
            f"indicating a {verdict} outlook.",
        ]

        if oracle.top_councils:
            best = oracle.top_councils[0]
            name = best.council_name or f"Council {best.council_id}"
            parts.append(
                f"The highest-affinity council is {name} "
                f"(score {best.score:.2f})."
            )

        return " ".join(parts)
