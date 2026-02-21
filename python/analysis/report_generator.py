"""PDF report generator — produces a professional due diligence document.

Takes a ranked list of :class:`~analysis.models.SiteViabilityReport` objects
and renders a multi-page PDF suitable for client delivery.

Sections:
    1. Cover page with title and generation date
    2. Executive summary with recommendation
    3. Per-borough detail pages:
       - Viability score gauge and breakdown
       - Approval prediction with risk / positive factors
       - Top 3 comparable applications table
       - Constraint flags summary
       - Estimated decision timeline
    4. Methodology note
    5. Legal disclaimer

Usage::

    from analysis.report_generator import generate_report

    pdf_bytes = generate_report(reports, query="20-unit housing in South London")
    Path("report.pdf").write_bytes(pdf_bytes)
"""

from __future__ import annotations

import io
from datetime import date
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from .models import ComparableApplication, SiteViabilityReport

__all__ = ["generate_report"]

# ---------------------------------------------------------------------------
# Brand colours
# ---------------------------------------------------------------------------

NAVY = colors.HexColor("#1B2A4A")
DARK_GREY = colors.HexColor("#2D3748")
MID_GREY = colors.HexColor("#4A5568")
LIGHT_GREY = colors.HexColor("#E2E8F0")
SURFACE = colors.HexColor("#F7FAFC")
WHITE = colors.white
GREEN = colors.HexColor("#2F855A")
AMBER = colors.HexColor("#C05621")
RED = colors.HexColor("#C53030")
ACCENT = colors.HexColor("#3182CE")
ACCENT_LIGHT = colors.HexColor("#EBF4FF")

PAGE_W, PAGE_H = A4
MARGIN = 22 * mm


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------


def _build_styles() -> dict[str, ParagraphStyle]:
    """Build the custom style sheet."""
    base = getSampleStyleSheet()

    return {
        # Cover page
        "cover_title": ParagraphStyle(
            "cover_title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=28,
            leading=34,
            textColor=WHITE,
            alignment=TA_LEFT,
        ),
        "cover_subtitle": ParagraphStyle(
            "cover_subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=14,
            leading=20,
            textColor=colors.HexColor("#CBD5E0"),
            alignment=TA_LEFT,
        ),
        "cover_date": ParagraphStyle(
            "cover_date",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=11,
            textColor=colors.HexColor("#A0AEC0"),
            alignment=TA_LEFT,
        ),
        # Body
        "h1": ParagraphStyle(
            "h1",
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=24,
            textColor=NAVY,
            spaceBefore=6,
            spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "h2",
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=DARK_GREY,
            spaceBefore=14,
            spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "h3",
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=15,
            textColor=MID_GREY,
            spaceBefore=10,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=DARK_GREY,
            spaceAfter=6,
        ),
        "body_small": ParagraphStyle(
            "body_small",
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=MID_GREY,
            spaceAfter=4,
        ),
        "body_bold": ParagraphStyle(
            "body_bold",
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=14,
            textColor=DARK_GREY,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=DARK_GREY,
            leftIndent=14,
            bulletIndent=0,
            spaceAfter=3,
        ),
        "score_large": ParagraphStyle(
            "score_large",
            fontName="Helvetica-Bold",
            fontSize=36,
            leading=40,
            alignment=TA_CENTER,
        ),
        "score_label": ParagraphStyle(
            "score_label",
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=MID_GREY,
            alignment=TA_CENTER,
        ),
        "table_header": ParagraphStyle(
            "table_header",
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=WHITE,
        ),
        "table_cell": ParagraphStyle(
            "table_cell",
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=DARK_GREY,
        ),
        "table_cell_bold": ParagraphStyle(
            "table_cell_bold",
            fontName="Helvetica-Bold",
            fontSize=8.5,
            leading=11,
            textColor=DARK_GREY,
        ),
        "disclaimer": ParagraphStyle(
            "disclaimer",
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=MID_GREY,
            spaceAfter=4,
        ),
        "footer": ParagraphStyle(
            "footer",
            fontName="Helvetica",
            fontSize=7.5,
            textColor=MID_GREY,
            alignment=TA_CENTER,
        ),
        "recommendation_tag": ParagraphStyle(
            "recommendation_tag",
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=WHITE,
        ),
    }


# ---------------------------------------------------------------------------
# Score colour helpers
# ---------------------------------------------------------------------------


def _band_colour(score: int) -> colors.HexColor:
    if score >= 70:
        return GREEN
    if score >= 40:
        return AMBER
    return RED


def _band_label(score: int) -> str:
    if score >= 70:
        return "HIGH"
    if score >= 40:
        return "MODERATE"
    return "LOW"


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.7:
        return "High confidence"
    if confidence >= 0.4:
        return "Moderate confidence"
    return "Low confidence"


# ---------------------------------------------------------------------------
# Page decorations
# ---------------------------------------------------------------------------


def _cover_page_decorator(canvas, doc):
    """Dark navy cover with accent stripe."""
    canvas.saveState()
    # Full-page navy background
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=True, stroke=False)
    # Accent stripe at bottom
    canvas.setFillColor(ACCENT)
    canvas.rect(0, 0, PAGE_W, 8 * mm, fill=True, stroke=False)
    # Thin accent line near top
    canvas.setStrokeColor(ACCENT)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, PAGE_H - 30 * mm, PAGE_W - MARGIN, PAGE_H - 30 * mm)
    canvas.restoreState()


def _body_page_decorator(canvas, doc):
    """Clean body page with header rule and footer."""
    canvas.saveState()
    # Header rule
    canvas.setStrokeColor(LIGHT_GREY)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, PAGE_H - 18 * mm, PAGE_W - MARGIN, PAGE_H - 18 * mm)
    # Header text
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MID_GREY)
    canvas.drawString(MARGIN, PAGE_H - 15 * mm, "Planning Due Diligence Report")
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 15 * mm, "Confidential")
    # Footer
    canvas.setStrokeColor(LIGHT_GREY)
    canvas.line(MARGIN, 16 * mm, PAGE_W - MARGIN, 16 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MID_GREY)
    canvas.drawCentredString(PAGE_W / 2, 10 * mm, f"Page {doc.page}")
    canvas.drawRightString(
        PAGE_W - MARGIN, 10 * mm, f"Generated {date.today().strftime('%d %B %Y')}"
    )
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_cover(
    story: list,
    styles: dict,
    query: str,
    report_count: int,
) -> None:
    """Cover page elements."""
    story.append(Spacer(1, 60 * mm))
    story.append(
        Paragraph("Planning Due Diligence<br/>Report", styles["cover_title"])
    )
    story.append(Spacer(1, 8 * mm))
    if query:
        story.append(Paragraph(f"&ldquo;{query}&rdquo;", styles["cover_subtitle"]))
        story.append(Spacer(1, 4 * mm))
    story.append(
        Paragraph(
            f"{report_count} borough{'s' if report_count != 1 else ''} analysed",
            styles["cover_subtitle"],
        )
    )
    story.append(Spacer(1, 12 * mm))
    story.append(
        Paragraph(
            f"Prepared {date.today().strftime('%d %B %Y')}",
            styles["cover_date"],
        )
    )
    story.append(NextPageTemplate("body"))
    story.append(PageBreak())


def _build_executive_summary(
    story: list,
    styles: dict,
    reports: list[SiteViabilityReport],
) -> None:
    """Executive summary with recommendation."""
    story.append(Paragraph("Executive Summary", styles["h1"]))
    story.append(
        HRFlowable(
            width="100%", thickness=1, color=ACCENT, spaceBefore=0, spaceAfter=8
        )
    )

    if not reports:
        story.append(Paragraph("No boroughs could be analysed.", styles["body"]))
        return

    top = reports[0]
    story.append(
        Paragraph(
            f"<b>Recommendation:</b> {top.borough} ranks highest with a viability "
            f"score of <b>{top.viability_score}/100</b> "
            f"({_band_label(top.viability_score).lower()} viability).",
            styles["body"],
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(top.summary, styles["body"]))
    story.append(Spacer(1, 4 * mm))

    # Summary table of all boroughs
    header = [
        Paragraph("Rank", styles["table_header"]),
        Paragraph("Borough", styles["table_header"]),
        Paragraph("Viability", styles["table_header"]),
        Paragraph("Approval Score", styles["table_header"]),
        Paragraph("Avg Decision", styles["table_header"]),
        Paragraph("Data Quality", styles["table_header"]),
    ]
    data = [header]
    for r in reports:
        weeks_str = (
            f"{r.borough_stats.avg_decision_weeks:.0f} wks"
            if r.borough_stats.avg_decision_weeks
            else "—"
        )
        data.append([
            Paragraph(f"#{r.rank}", styles["table_cell_bold"]),
            Paragraph(r.borough, styles["table_cell_bold"]),
            Paragraph(f"{r.viability_score}/100", styles["table_cell"]),
            Paragraph(f"{r.approval_prediction.score}/100", styles["table_cell"]),
            Paragraph(weeks_str, styles["table_cell"]),
            Paragraph(r.data_quality.upper(), styles["table_cell"]),
        ])

    col_widths = [30, 100, 55, 75, 65, 60]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, SURFACE]),
        ("GRID", (0, 0), (-1, -1), 0.5, LIGHT_GREY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 6 * mm))


def _build_borough_detail(
    story: list,
    styles: dict,
    report: SiteViabilityReport,
) -> None:
    """Full detail page for a single borough."""
    # Borough header with recommendation badge
    title = f"{report.borough}"
    if report.recommended:
        title += '&nbsp;&nbsp;<font color="#2F855A">&#x2713; RECOMMENDED</font>'
    story.append(Paragraph(title, styles["h1"]))
    story.append(
        HRFlowable(
            width="100%", thickness=1, color=ACCENT, spaceBefore=0, spaceAfter=6
        )
    )

    # ── Score cards row ──────────────────────────────────────────────
    score_colour = _band_colour(report.viability_score)
    pred_colour = _band_colour(report.approval_prediction.score)

    score_cards_data = [[
        _score_card(
            report.viability_score,
            "VIABILITY SCORE",
            score_colour,
            styles,
        ),
        _score_card(
            report.approval_prediction.score,
            "APPROVAL LIKELIHOOD",
            pred_colour,
            styles,
        ),
        _score_card(
            int(report.borough_stats.approval_rate),
            "BOROUGH APPROVAL %",
            ACCENT,
            styles,
            suffix="%",
        ),
        _stat_card(
            f"{report.borough_stats.avg_decision_weeks:.0f} wks"
            if report.borough_stats.avg_decision_weeks
            else "—",
            "AVG DECISION TIME",
            styles,
        ),
    ]]
    card_table = Table(score_cards_data, colWidths=[95, 95, 95, 95])
    card_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(card_table)
    story.append(Spacer(1, 6 * mm))

    # ── Summary ──────────────────────────────────────────────────────
    story.append(Paragraph(report.summary, styles["body"]))
    story.append(Spacer(1, 2 * mm))

    # ── Prediction verdict ───────────────────────────────────────────
    pred = report.approval_prediction
    story.append(
        Paragraph(
            f"<b>Verdict:</b> {pred.verdict} "
            f"({_confidence_label(pred.confidence)})",
            styles["body"],
        )
    )

    # ── Risk & positive factors ──────────────────────────────────────
    if pred.risk_factors or pred.positive_factors:
        story.append(Paragraph("Risk &amp; Opportunity Factors", styles["h2"]))
        _build_factors_table(story, styles, pred.risk_factors, pred.positive_factors)

    # ── Constraint flags ─────────────────────────────────────────────
    story.append(Paragraph("Planning Constraints", styles["h2"]))
    _build_constraints(story, styles, report.constraint_flags)

    # ── Comparable applications ──────────────────────────────────────
    comps = report.comparable_applications[:3]
    if comps:
        story.append(Paragraph("Top Comparable Applications", styles["h2"]))
        _build_comparables_table(story, styles, comps)

    # ── Key considerations ───────────────────────────────────────────
    if report.key_considerations:
        story.append(Paragraph("Key Considerations", styles["h2"]))
        for item in report.key_considerations:
            story.append(
                Paragraph(f"&bull;&nbsp;&nbsp;{item}", styles["bullet"])
            )

    # ── Decision timeline ────────────────────────────────────────────
    story.append(Paragraph("Estimated Decision Timeline", styles["h2"]))
    _build_timeline(story, styles, report)


def _score_card(
    value: int,
    label: str,
    colour: colors.HexColor,
    styles: dict,
    suffix: str = "",
) -> list:
    """Render a single score card as a nested table."""
    display = f"{value}{suffix}"
    inner = [
        [Paragraph(
            f'<font color="{colour.hexval()}">{display}</font>',
            styles["score_large"],
        )],
        [Paragraph(label, styles["score_label"])],
    ]
    t = Table(inner, colWidths=[88])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SURFACE),
        ("BOX", (0, 0), (-1, -1), 0.5, LIGHT_GREY),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def _stat_card(
    value: str,
    label: str,
    styles: dict,
) -> list:
    """Render a text stat card."""
    stat_style = ParagraphStyle(
        "stat_value",
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=26,
        textColor=DARK_GREY,
        alignment=TA_CENTER,
    )
    inner = [
        [Paragraph(value, stat_style)],
        [Paragraph(label, styles["score_label"])],
    ]
    t = Table(inner, colWidths=[88])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SURFACE),
        ("BOX", (0, 0), (-1, -1), 0.5, LIGHT_GREY),
        ("TOPPADDING", (0, 0), (-1, 0), 12),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def _build_factors_table(
    story: list,
    styles: dict,
    risks: list,
    positives: list,
) -> None:
    """Render risk and positive factors as a compact table."""
    header = [
        Paragraph("Factor", styles["table_header"]),
        Paragraph("Impact", styles["table_header"]),
        Paragraph("Type", styles["table_header"]),
        Paragraph("Detail", styles["table_header"]),
    ]
    data = [header]

    for f in risks[:4]:
        data.append([
            Paragraph(f.label, styles["table_cell_bold"]),
            Paragraph(
                f'<font color="{RED.hexval()}">{f.score_impact:+d}</font>',
                styles["table_cell"],
            ),
            Paragraph(f.category.title(), styles["table_cell"]),
            Paragraph(f.description, styles["table_cell"]),
        ])

    for f in positives[:3]:
        data.append([
            Paragraph(f.label, styles["table_cell_bold"]),
            Paragraph(
                f'<font color="{GREEN.hexval()}">{f.score_impact:+d}</font>',
                styles["table_cell"],
            ),
            Paragraph(f.category.title(), styles["table_cell"]),
            Paragraph(f.description, styles["table_cell"]),
        ])

    if len(data) == 1:
        story.append(Paragraph("No significant risk factors identified.", styles["body"]))
        return

    t = Table(data, colWidths=[100, 45, 60, 180], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, SURFACE]),
        ("GRID", (0, 0), (-1, -1), 0.5, LIGHT_GREY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)


def _build_constraints(story: list, styles: dict, flags) -> None:
    """Render constraint flags as a compact grid."""
    active = flags.active_flags
    if not active:
        story.append(Paragraph("No significant constraints detected.", styles["body"]))
        return

    # Two-column layout
    labels = [f.replace("_", " ").title() for f in active]
    half = (len(labels) + 1) // 2
    col1 = labels[:half]
    col2 = labels[half:]

    data = []
    for i in range(max(len(col1), len(col2))):
        row = []
        if i < len(col1):
            row.append(Paragraph(
                f'<font color="{RED.hexval()}">&#x2716;</font>&nbsp;&nbsp;{col1[i]}',
                styles["body_small"],
            ))
        else:
            row.append(Paragraph("", styles["body_small"]))
        if i < len(col2):
            row.append(Paragraph(
                f'<font color="{RED.hexval()}">&#x2716;</font>&nbsp;&nbsp;{col2[i]}',
                styles["body_small"],
            ))
        else:
            row.append(Paragraph("", styles["body_small"]))
        data.append(row)

    t = Table(data, colWidths=[190, 190])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)

    story.append(
        Paragraph(
            f"Data source: <i>{flags.data_source}</i>",
            styles["body_small"],
        )
    )


def _build_comparables_table(
    story: list,
    styles: dict,
    comps: list[ComparableApplication],
) -> None:
    """Render the top comparable applications."""
    header = [
        Paragraph("Reference", styles["table_header"]),
        Paragraph("Decision", styles["table_header"]),
        Paragraph("Score", styles["table_header"]),
        Paragraph("Proposal", styles["table_header"]),
        Paragraph("Relevance", styles["table_header"]),
    ]
    data = [header]

    for c in comps:
        decision_colour = GREEN if c.normalised_decision.value == "Approved" else RED
        # Truncate proposal
        proposal_text = (c.proposal or "—")[:80]
        if c.proposal and len(c.proposal) > 80:
            proposal_text += "..."
        # Top 2 reasons
        reasons = c.similarity_reasons[:2]
        reasons_text = "; ".join(reasons) if reasons else "—"

        data.append([
            Paragraph(c.planning_reference, styles["table_cell_bold"]),
            Paragraph(
                f'<font color="{decision_colour.hexval()}">'
                f'{c.normalised_decision.value}</font>',
                styles["table_cell"],
            ),
            Paragraph(f"{c.similarity_score:.0%}", styles["table_cell"]),
            Paragraph(proposal_text, styles["table_cell"]),
            Paragraph(reasons_text, styles["table_cell"]),
        ])

    t = Table(data, colWidths=[72, 52, 36, 130, 95], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, SURFACE]),
        ("GRID", (0, 0), (-1, -1), 0.5, LIGHT_GREY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)


def _build_timeline(
    story: list,
    styles: dict,
    report: SiteViabilityReport,
) -> None:
    """Render a simple timeline estimate."""
    weeks = report.borough_stats.avg_decision_weeks
    if weeks is None:
        story.append(
            Paragraph(
                "Insufficient data to estimate decision timeline.",
                styles["body"],
            )
        )
        return

    phases = [
        ("Pre-application advice", "2–4 weeks"),
        ("Application preparation", "4–8 weeks"),
        ("Validation period", "1–2 weeks"),
        ("Consultation & assessment", f"{max(4, int(weeks) - 4)}–{int(weeks)} weeks"),
        ("Decision", f"~{int(weeks)} weeks from submission"),
    ]

    data = []
    for i, (phase, duration) in enumerate(phases):
        marker_colour = ACCENT if i < len(phases) - 1 else GREEN
        data.append([
            Paragraph(
                f'<font color="{marker_colour.hexval()}">&#x25CF;</font>',
                styles["body_small"],
            ),
            Paragraph(f"<b>{phase}</b>", styles["body_small"]),
            Paragraph(duration, styles["body_small"]),
        ])

    t = Table(data, colWidths=[16, 160, 180])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(t)


def _build_methodology(story: list, styles: dict) -> None:
    """Methodology section."""
    story.append(PageBreak())
    story.append(Paragraph("Methodology", styles["h1"]))
    story.append(
        HRFlowable(
            width="100%", thickness=1, color=ACCENT, spaceBefore=0, spaceAfter=8
        )
    )

    paras = [
        (
            "This report is generated using a statistical analysis engine that "
            "processes real planning application data from the IBex Enterprise API, "
            "covering all local planning authorities in England and Wales."
        ),
        (
            "<b>Viability score</b> is a weighted composite of four dimensions: "
            "approval prediction (50%), comparable precedent evidence (20%), "
            "decision speed (15%), and borough development activity (15%)."
        ),
        (
            "<b>Approval prediction</b> blends the borough-wide approval rate with "
            "the outcome of comparable applications, adjusted for planning constraint "
            "penalties (conservation area, flood zone, Green Belt, Article 4)."
        ),
        (
            "<b>Comparable selection</b> uses a six-dimensional similarity score: "
            "application type match (30%), borough match (25%), unit count proximity "
            "(20%), project type (10%), recency (10%), and keyword overlap (5%). "
            "Only decided applications (Approved or Refused) are used as precedents."
        ),
        (
            "<b>Constraint inference</b> is derived heuristically from application "
            "metadata and proposal text when GIS constraint layers are unavailable. "
            "Flood zone and Green Belt flags require external GIS data for accuracy."
        ),
        (
            "<b>Data quality</b> is classified as: Full (10+ decided applications), "
            "Partial (1–9 decided), or Mock (no real data — synthetic statistics are "
            "used as placeholders)."
        ),
    ]

    for text in paras:
        story.append(Paragraph(text, styles["body"]))
    story.append(Spacer(1, 4 * mm))


def _build_disclaimer(story: list, styles: dict) -> None:
    """Legal disclaimer."""
    story.append(Paragraph("Disclaimer", styles["h2"]))
    story.append(
        HRFlowable(
            width="100%", thickness=0.5, color=LIGHT_GREY, spaceBefore=0, spaceAfter=6
        )
    )

    paras = [
        (
            "This report is produced for informational purposes only and does not "
            "constitute legal, financial, or professional planning advice. The "
            "analysis is based on statistical modelling of publicly available "
            "planning application data and should not be relied upon as a guarantee "
            "of any planning outcome."
        ),
        (
            "Planning decisions are made by local planning authorities on a "
            "case-by-case basis, taking into account material considerations that "
            "may not be captured in historical application data. Site-specific "
            "factors — including neighbour objections, design quality, ecology, "
            "highways impact, and officer discretion — can materially affect "
            "outcomes."
        ),
        (
            "Recipients should obtain independent professional advice from a "
            "qualified planning consultant before making any investment, "
            "acquisition, or development decisions based on the contents of this "
            "report. The authors accept no liability for any loss or damage "
            "arising from reliance on this analysis."
        ),
        (
            "Data is sourced from the IBex Enterprise API. While every effort "
            "is made to ensure accuracy, the underlying data may contain errors, "
            "omissions, or delays in reporting. Constraint flags derived from "
            "heuristic analysis may not reflect the complete planning policy "
            "context for a given site."
        ),
    ]

    for text in paras:
        story.append(Paragraph(text, styles["disclaimer"]))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_report(
    reports: list[SiteViabilityReport],
    *,
    query: str = "",
) -> bytes:
    """Generate a professional PDF due diligence report.

    Args:
        reports: Ranked list of :class:`SiteViabilityReport` objects
                 (rank 1 first).
        query:   The original natural language query (shown on the cover).

    Returns:
        PDF file contents as bytes.
    """
    buf = io.BytesIO()
    styles = _build_styles()

    # Page templates
    cover_frame = Frame(
        MARGIN, MARGIN, PAGE_W - 2 * MARGIN, PAGE_H - 2 * MARGIN,
        id="cover",
    )
    body_frame = Frame(
        MARGIN, 20 * mm, PAGE_W - 2 * MARGIN, PAGE_H - 42 * mm,
        id="body",
    )

    doc = BaseDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=24 * mm,
        bottomMargin=20 * mm,
    )
    doc.addPageTemplates([
        PageTemplate(
            id="cover",
            frames=[cover_frame],
            onPage=_cover_page_decorator,
        ),
        PageTemplate(
            id="body",
            frames=[body_frame],
            onPage=_body_page_decorator,
        ),
    ])

    story: list = []

    # 1. Cover
    _build_cover(story, styles, query, len(reports))

    # 2. Executive summary
    _build_executive_summary(story, styles, reports)

    # 3. Per-borough detail
    for i, report in enumerate(reports):
        story.append(PageBreak())
        _build_borough_detail(story, styles, report)

    # 4. Methodology + Disclaimer
    _build_methodology(story, styles)
    _build_disclaimer(story, styles)

    doc.build(story)
    return buf.getvalue()
