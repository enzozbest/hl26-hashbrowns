"""
Standalone PDF generation test with fixture data.

Usage:
    cd python
    python scripts/test_pdf_generation.py
    python scripts/test_pdf_generation.py --output /tmp/my_report.pdf
    python scripts/test_pdf_generation.py --scenario high_viability
    python scripts/test_pdf_generation.py --scenario low_viability
    python scripts/test_pdf_generation.py --scenario mixed
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

# Make sure the python/ directory is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.models import (
    ApprovalPrediction,
    BoroughStats,
    ComparableApplication,
    ConstraintFlags,
    RiskFactor,
    SiteViabilityReport,
)
from analysis.report_generator import generate_report

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_comparable(
    ref: str,
    address: str,
    description: str,
    decision: str,
    units: int,
    reason: str,
    weeks: int = 14,
) -> ComparableApplication:
    return ComparableApplication(
        planning_reference=ref,
        council_name=address.split(",")[-1].strip(),   # derive from address
        address=address,
        description=description,
        normalised_decision=decision,
        decision_date=date(2023, 6, 15),
        units=units,
        application_type="full planning application",
        similarity_reason=reason,
        decision_weeks=weeks,
        url=f"https://ibex.example.com/applications/{ref}",
    )


def _make_report(
    borough: str,
    rank: int,
    viability_score: int,
    viability_band: str,
    approval_score: int,
    confidence: str,
    verdict: str,
    approval_rate: float,
    avg_weeks: float,
    trend: str,
    constraint_flags: ConstraintFlags,
    penalties: list[tuple[str, int]],
    comparables: list[ComparableApplication],
    data_quality: str = "full",
) -> SiteViabilityReport:
    risk_factors = [
        RiskFactor(label=label, score_impact=-penalty, description=f"{label} constraint detected")
        for label, penalty in penalties
    ]
    positive_factors = [
        RiskFactor(label="Strong approval history", score_impact=10,
                   description=f"{approval_rate:.0f}% approval rate in this borough"),
    ]

    prediction = ApprovalPrediction(
        score=approval_score,
        confidence=confidence,
        comparable_approval_rate=72.0 if comparables else None,
        num_comparables=len(comparables),
        borough_baseline_rate=approval_rate,
        risk_factors=risk_factors,
        positive_factors=positive_factors,
        verdict=verdict,
    )

    stats = BoroughStats(
        name=borough,
        council_id=None,
        total_applications=120,
        approved=round(120 * approval_rate / 100),
        refused=round(120 * (100 - approval_rate) / 100),
        pending=5,
        withdrawn=3,
        approval_rate=approval_rate,
        refusal_rate=round(100 - approval_rate, 1),
        avg_decision_weeks=avg_weeks,
        trend=trend,
        trend_detail=f"{approval_rate:.0f}% last 2 yrs vs {approval_rate - 5:.0f}% prior",
        data_quality=data_quality,
        comparable_count=len(comparables),
    )

    return SiteViabilityReport(
        borough=borough,
        council_id=None,
        rank=rank,
        viability_score=viability_score,
        viability_band=viability_band,
        approval_prediction=prediction,
        borough_stats=stats,
        constraint_flags=constraint_flags,
        comparable_applications=comparables,
        summary=(
            f"{borough} shows {'strong' if viability_score >= 70 else 'moderate' if viability_score >= 40 else 'weak'} "
            f"viability for a 20-unit affordable housing scheme. "
            f"Historical approval rate is {approval_rate:.0f}% with an average decision time of {avg_weeks:.0f} weeks."
        ),
        key_considerations=[
            f"Average decision time: {avg_weeks:.0f} weeks",
            f"Approval rate: {approval_rate:.0f}% (all types)",
            *(f"{label} constraint applies (−{pts} pts)" for label, pts in penalties),
            f"Trend: {trend}",
        ],
        data_quality=data_quality,
    )


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

_COMPARABLES_LAMBETH = [
    _make_comparable("LB/23/001", "45 Brixton Rd, Lambeth", "20 affordable units", "Approved", 20,
                     "Exact unit count match, same borough, full planning", 12),
    _make_comparable("LB/22/087", "12 Stockwell Ave, Lambeth", "18 affordable flats", "Approved", 18,
                     "Similar scale (18 units), affordable housing, same borough", 15),
    _make_comparable("LB/23/044", "8 Clapham High St", "22 residential units", "Refused", 22,
                     "Similar scale (22 units), refused — design objections", 20),
]

_COMPARABLES_SOUTHWARK = [
    _make_comparable("SW/23/112", "Old Kent Rd, Southwark", "20 mixed-tenure units", "Approved", 20,
                     "Exact unit count, full planning, same borough", 11),
    _make_comparable("SW/22/055", "Peckham High St", "16 affordable flats", "Approved", 16,
                     "Affordable housing focus, similar scale", 14),
    _make_comparable("SW/21/203", "Bermondsey St", "25 units", "Refused", 25,
                     "Larger scheme refused — height objections", 18),
]

_COMPARABLES_CROYDON = [
    _make_comparable("CR/23/009", "North End, Croydon", "20 units affordable", "Approved", 20,
                     "Exact match on unit count and type", 16),
    _make_comparable("CR/22/078", "George St, Croydon", "15 affordable flats", "Refused", 15,
                     "Smaller affordable scheme refused — viability concerns", 22),
]


SCENARIOS: dict[str, list[SiteViabilityReport]] = {
    "high_viability": [
        _make_report(
            "Lambeth", 1, 82, "High", 85, "high",
            "Strong approval likelihood based on comparable precedents and favourable borough trends.",
            82.0, 11.5, "improving",
            ConstraintFlags(data_source="heuristic"),
            [],
            _COMPARABLES_LAMBETH,
        ),
        _make_report(
            "Southwark", 2, 74, "High", 76, "high",
            "Good viability. Southwark has a strong track record for affordable housing.",
            78.0, 13.0, "stable",
            ConstraintFlags(article_4=True, data_source="heuristic"),
            [("Article 4", 10)],
            _COMPARABLES_SOUTHWARK,
        ),
        _make_report(
            "Croydon", 3, 61, "Medium", 63, "medium",
            "Moderate viability. Croydon approves most affordable schemes but decision times are longer.",
            68.0, 17.0, "stable",
            ConstraintFlags(data_source="heuristic"),
            [],
            _COMPARABLES_CROYDON,
        ),
    ],
    "low_viability": [
        _make_report(
            "Kingston upon Thames", 1, 38, "Low", 35, "low",
            "Challenging environment. Conservation constraints and declining approval trend reduce viability.",
            55.0, 19.0, "declining",
            ConstraintFlags(conservation_area=True, article_4=True, data_source="heuristic"),
            [("Conservation area", 20), ("Article 4", 10)],
            [],
            data_quality="partial",
        ),
        _make_report(
            "Richmond upon Thames", 2, 28, "Low", 25, "low",
            "Low viability. Green Belt and conservation area constraints significantly limit development.",
            60.0, 21.0, "declining",
            ConstraintFlags(green_belt=True, conservation_area=True, data_source="heuristic"),
            [("Green Belt", 25), ("Conservation area", 20)],
            [],
            data_quality="partial",
        ),
    ],
    "mixed": [
        _make_report(
            "Lambeth", 1, 78, "High", 80, "high",
            "Strong viability for 20-unit affordable scheme. Lambeth has a proactive affordable housing policy.",
            80.0, 12.0, "improving",
            ConstraintFlags(data_source="heuristic"),
            [],
            _COMPARABLES_LAMBETH,
        ),
        _make_report(
            "Lewisham", 2, 55, "Medium", 57, "medium",
            "Moderate viability. Lewisham approves most residential schemes but Article 4 adds risk.",
            65.0, 15.0, "stable",
            ConstraintFlags(article_4=True, data_source="heuristic"),
            [("Article 4", 10)],
            [],
        ),
        _make_report(
            "Bromley", 3, 32, "Low", 30, "low",
            "Low viability. Green Belt coverage across much of the borough restricts development.",
            58.0, 20.0, "declining",
            ConstraintFlags(green_belt=True, flood_risk=True, data_source="heuristic"),
            [("Green Belt", 25), ("Flood zone", 15)],
            [],
            data_quality="partial",
        ),
    ],
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate test PDFs for the due diligence report pipeline.")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()), default="mixed",
                        help="Which fixture scenario to render (default: mixed)")
    parser.add_argument("--output", default=None,
                        help="Output PDF path (default: /tmp/dd_report_<scenario>.pdf)")
    parser.add_argument("--all", action="store_true",
                        help="Generate PDFs for all scenarios")
    args = parser.parse_args()

    scenarios_to_run = list(SCENARIOS.keys()) if args.all else [args.scenario]

    for scenario_name in scenarios_to_run:
        reports = SCENARIOS[scenario_name]
        out_path = args.output or f"/tmp/dd_report_{scenario_name}.pdf"

        print(f"\n→ Generating '{scenario_name}' scenario ({len(reports)} boroughs)...")
        pdf_bytes = generate_report(reports)

        Path(out_path).write_bytes(pdf_bytes)
        size_kb = len(pdf_bytes) // 1024
        print(f"  ✓ Written to {out_path} ({size_kb} KB)")

        if size_kb < 50:
            print(f"  ⚠️  WARNING: PDF is only {size_kb} KB — may be incomplete")
        else:
            print(f"  ✓ Size looks healthy (> 50 KB)")

        # Quick section heading check
        text_preview = pdf_bytes[:4096]
        if b"Executive Summary" in pdf_bytes or b"EXECUTIVE" in pdf_bytes:
            print("  ✓ Executive Summary section detected")
        else:
            print("  ⚠️  Executive Summary heading not found in PDF bytes")


if __name__ == "__main__":
    main()