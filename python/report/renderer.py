"""PDF renderer for CouncilReport.

Produces a professional, multi-page PDF using ReportLab.  Each section in
the report is rendered as a self-contained block; adding a new section
requires no changes here.

Usage::

    from report.builder import build_report
    from report.renderer import render_pdf

    pdf_bytes = render_pdf(build_report("Hackney"))
    Path("hackney_report.pdf").write_bytes(pdf_bytes)
"""

from __future__ import annotations

import io
from datetime import datetime

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

from report.models import CouncilReport, Insight, Metric, SectionResult

__all__ = ["render_pdf"]

# ---------------------------------------------------------------------------
# Palette  — earthy dark theme matching the frontend
# ---------------------------------------------------------------------------

PAGE_BG  = colors.HexColor("#1e1b17")   # warm charcoal background
CARD_BG  = colors.HexColor("#26211b")   # slightly raised surface
BORDER   = colors.HexColor("#4a3c20")   # dark gold rule
ACCENT   = colors.HexColor("#c8962a")   # ochre gold
TEXT     = colors.HexColor("#ede8df")   # warm off-white
MUTED    = colors.HexColor("#a89b86")   # dimmed text
FAINT    = colors.HexColor("#5c5344")   # very dim text / source lines
GREEN    = colors.HexColor("#6b8f5e")   # positive (matches frontend riskMeta)
AMBER    = colors.HexColor("#c8962a")   # moderate — same as accent
RED      = colors.HexColor("#b85c38")   # negative / low approval
WHITE    = colors.HexColor("#ede8df")   # reuse TEXT as "white" for covers

SENTIMENT_COLOR = {
    "positive": GREEN,
    "negative": RED,
    "neutral":  MUTED,
}

DIRECTION_COLOR = {
    "positive": GREEN,
    "negative": RED,
    "neutral":  TEXT,
}

PAGE_W, PAGE_H = A4
MARGIN = 20 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()

    def s(name, **kw) -> ParagraphStyle:
        parent = kw.pop("parent", "Normal")
        return ParagraphStyle(name, parent=base[parent], **kw)

    return {
        "cover_title": s(
            "cover_title",
            fontSize=32,
            leading=38,
            textColor=ACCENT,
            fontName="Helvetica-Bold",
            alignment=TA_LEFT,
        ),
        "cover_sub": s(
            "cover_sub",
            fontSize=12,
            leading=17,
            textColor=TEXT,
            fontName="Helvetica",
            alignment=TA_LEFT,
        ),
        "cover_meta": s(
            "cover_meta",
            fontSize=8.5,
            leading=13,
            textColor=MUTED,
            fontName="Helvetica",
            alignment=TA_LEFT,
        ),
        "section_heading": s(
            "section_heading",
            fontSize=13,
            leading=17,
            textColor=ACCENT,
            fontName="Helvetica-Bold",
            spaceAfter=2 * mm,
        ),
        "body": s(
            "body",
            fontSize=9,
            leading=14,
            textColor=TEXT,
            fontName="Helvetica",
            spaceAfter=3 * mm,
        ),
        "label": s(
            "label",
            fontSize=7.5,
            leading=10,
            textColor=MUTED,
            fontName="Helvetica",
        ),
        "metric_value": s(
            "metric_value",
            fontSize=15,
            leading=18,
            textColor=TEXT,
            fontName="Helvetica-Bold",
        ),
        "metric_context": s(
            "metric_context",
            fontSize=7,
            leading=10,
            textColor=MUTED,
            fontName="Helvetica-Oblique",
        ),
        "insight": s(
            "insight",
            fontSize=8.5,
            leading=13,
            textColor=TEXT,
            fontName="Helvetica",
        ),
        "footer": s(
            "footer",
            fontSize=7,
            leading=10,
            textColor=FAINT,
            fontName="Helvetica",
            alignment=TA_CENTER,
        ),
        "disclaimer": s(
            "disclaimer",
            fontSize=7.5,
            leading=11,
            textColor=MUTED,
            fontName="Helvetica-Oblique",
        ),
    }


# ---------------------------------------------------------------------------
# Page templates
# ---------------------------------------------------------------------------


class _Doc(BaseDocTemplate):
    def __init__(self, buf: io.BytesIO, report: CouncilReport, st: dict):
        self._report = report
        self._st = st
        super().__init__(
            buf,
            pagesize=A4,
            leftMargin=MARGIN,
            rightMargin=MARGIN,
            topMargin=MARGIN,
            bottomMargin=MARGIN + 8 * mm,
            title=f"Planning Intelligence Report — {report.council.council_name}",
            author="Hashbrowns Planning Intelligence",
        )
        self._build_templates()

    def _build_templates(self):
        cover_frame = Frame(MARGIN * 1.8, 0, PAGE_W - MARGIN * 1.8, PAGE_H, id="cover_frame", showBoundary=0)
        body_frame = Frame(
            MARGIN, MARGIN + 8 * mm,
            CONTENT_W, PAGE_H - 2 * MARGIN - 8 * mm,
            id="body_frame", showBoundary=0,
        )
        self.addPageTemplates([
            PageTemplate(id="Cover", frames=[cover_frame], onPage=_draw_cover_bg),
            PageTemplate(id="Body", frames=[body_frame], onPage=self._draw_body_chrome),
        ])

    def _draw_body_chrome(self, canvas, doc):
        canvas.saveState()
        # Dark page background
        canvas.setFillColor(PAGE_BG)
        canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        # Thin gold header rule
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, PAGE_H - MARGIN + 3 * mm, PAGE_W - MARGIN, PAGE_H - MARGIN + 3 * mm)
        # Council name top-left in accent gold
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(ACCENT)
        canvas.drawString(MARGIN, PAGE_H - MARGIN + 5 * mm, self._report.council.council_name)
        # Page number top-right in muted
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - MARGIN + 5 * mm, f"Page {doc.page}")
        # Footer rule
        canvas.setStrokeColor(BORDER)
        canvas.line(MARGIN, MARGIN + 6 * mm, PAGE_W - MARGIN, MARGIN + 6 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(FAINT)
        canvas.drawCentredString(
            PAGE_W / 2, MARGIN + 2 * mm,
            "Planning Intelligence Report — Clearance | Confidential"
        )
        canvas.restoreState()


def _draw_cover_bg(canvas, doc):
    canvas.saveState()
    # Dark warm background
    canvas.setFillColor(PAGE_BG)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    # Horizontal gold accent bar
    canvas.setFillColor(ACCENT)
    canvas.rect(0, PAGE_H * 0.38, PAGE_W, 2, fill=1, stroke=0)
    # Subtle left-edge vertical bar
    canvas.setFillColor(BORDER)
    canvas.rect(0, 0, 4, PAGE_H, fill=1, stroke=0)
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Cover page
# ---------------------------------------------------------------------------


def _cover_page(report: CouncilReport, st: dict) -> list:
    council = report.council
    generated = report.generated_at.strftime("%d %B %Y, %H:%M UTC")
    spacer_top = Spacer(1, PAGE_H * 0.42)

    title = Paragraph(council.council_name.upper(), st["cover_title"])
    sub = Paragraph("Planning Intelligence Report", st["cover_sub"])
    meta_lines = [
        f"Region: {council.region_name}",
        f"ONS Code: {council.ons_code}",
        f"Generated: {generated}",
    ]
    meta = Paragraph(" &nbsp;&nbsp;·&nbsp;&nbsp; ".join(meta_lines), st["cover_meta"])

    return [
        NextPageTemplate("Body"),
        spacer_top,
        Spacer(1, 4 * mm),
        title,
        Spacer(1, 3 * mm),
        sub,
        Spacer(1, 6 * mm),
        meta,
        PageBreak(),
    ]


# ---------------------------------------------------------------------------
# Section rendering
# ---------------------------------------------------------------------------


def _render_section(section: SectionResult, st: dict) -> list:
    """Render one SectionResult into a list of ReportLab flowables."""
    elems: list = []

    # Heading + rule
    elems.append(Paragraph(section.title, st["section_heading"]))
    elems.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=3 * mm))

    # Data quality badge
    if section.data_quality != "full":
        badge_color = AMBER if section.data_quality == "partial" else RED
        badge_text = f"Data quality: {section.data_quality.upper()}"
        badge = Table(
            [[Paragraph(badge_text, ParagraphStyle(
                "badge", fontSize=7, textColor=WHITE, fontName="Helvetica-Bold",
            ))]],
            colWidths=[40 * mm],
        )
        badge.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), badge_color),
            ("ROUNDEDCORNERS", [3]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))
        elems.append(badge)
        elems.append(Spacer(1, 3 * mm))

    # Summary paragraph
    elems.append(Paragraph(section.summary, st["body"]))

    # Metrics grid (3 per row)
    if section.metrics:
        elems += _render_metrics(section.metrics, st)
        elems.append(Spacer(1, 4 * mm))

    # Insights
    if section.insights:
        elems += _render_insights(section.insights, st)
        elems.append(Spacer(1, 3 * mm))

    # Attribution
    if section.data_source:
        elems.append(Paragraph(f"Source: {section.data_source}", st["disclaimer"]))

    elems.append(Spacer(1, 8 * mm))
    return elems


def _render_metrics(metrics: list[Metric], st: dict) -> list:
    COLS = 3
    cell_w = CONTENT_W / COLS

    rows = []
    for i in range(0, len(metrics), COLS):
        chunk = metrics[i:i + COLS]
        row_cells = []
        for m in chunk:
            value_str = f"{m.value}{m.unit}" if m.unit else str(m.value)
            color = DIRECTION_COLOR.get(m.direction or "neutral", TEXT)
            value_para = Paragraph(
                f'<font color="#{color.hexval()[2:]}">{value_str}</font>',
                st["metric_value"],
            )
            cell = [
                Paragraph(m.label, st["label"]),
                value_para,
                Paragraph(m.context or "", st["metric_context"]),
            ]
            row_cells.append(cell)
        # Pad to COLS with empty cells
        while len(row_cells) < COLS:
            row_cells.append([Paragraph("", st["label"])])
        rows.append(row_cells)

    all_rows = []
    for row in rows:
        table_data = [[cell for cell in row]]
        t = Table(table_data, colWidths=[cell_w] * COLS)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
            ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        all_rows.append(t)
        all_rows.append(Spacer(1, 1.5 * mm))

    return all_rows


def _render_insights(insights: list[Insight], st: dict) -> list:
    elems = []
    for insight in insights:
        color = SENTIMENT_COLOR.get(insight.sentiment, MUTED)
        bullet = Table(
            [[
                Paragraph(
                    f'<font color="#{color.hexval()[2:]}">&#9632;</font>',
                    ParagraphStyle("bul_icon", fontSize=8, leading=13, fontName="Helvetica"),
                ),
                Paragraph(insight.text, st["insight"]),
            ]],
            colWidths=[5 * mm, CONTENT_W - 5 * mm],
        )
        bullet.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ]))
        elems.append(bullet)
        elems.append(Spacer(1, 1.5 * mm))
    return elems


# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------


def _disclaimer(st: dict) -> list:
    text = (
        "This report is generated from public datasets and is provided for indicative "
        "purposes only. It does not constitute professional planning advice. Income "
        "figures are derived from ONS MSOA-level estimates and aggregated to local "
        "authority level; they may not reflect micro-level site conditions. Users "
        "should seek independent professional advice before making investment or "
        "development decisions."
    )
    return [
        HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=3 * mm),
        Paragraph("Disclaimer", ParagraphStyle(
            "disc_head", fontSize=9, fontName="Helvetica-Bold", textColor=MUTED,
        )),
        Spacer(1, 2 * mm),
        Paragraph(text, st["disclaimer"]),
    ]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_pdf(report: CouncilReport) -> bytes:
    """Render *report* to a PDF and return the raw bytes."""
    buf = io.BytesIO()
    st = _styles()
    doc = _Doc(buf, report, st)

    story = []
    story += _cover_page(report, st)

    for section in report.sections:
        story += _render_section(section, st)

    story += _disclaimer(st)

    doc.build(story)
    return buf.getvalue()
