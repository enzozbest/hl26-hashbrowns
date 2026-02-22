"""Quick script to generate a sample council report PDF.

Usage:
    uv run python -m report.generate
    uv run python -m report.generate "Richmond upon Thames"
    uv run python -m report.generate Hackney
"""

import sys
from datetime import date
from pathlib import Path

from report import build_report, render_pdf

council = sys.argv[1] if len(sys.argv) > 1 else "Hackney"

print(f"Generating report for '{council}'...")
report = build_report(council)
pdf = render_pdf(report)

slug = report.council.council_name.lower().replace(" ", "_")
today = date.today().strftime("%Y%m%d")
out = Path(__file__).parent / f"{slug}_{today}.pdf"
out.write_bytes(pdf)
print(f"Written: {out}")
