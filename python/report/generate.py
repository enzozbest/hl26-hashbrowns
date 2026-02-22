"""Quick script to generate a sample council report PDF.

Usage:
    uv run python -m report.generate
    uv run python -m report.generate "Richmond upon Thames"
    uv run python -m report.generate Hackney 2023
"""

import sys
from pathlib import Path

from report import build_report, render_pdf

council = sys.argv[1] if len(sys.argv) > 1 else "Hackney"
year = sys.argv[2] if len(sys.argv) > 2 else "2023"

print(f"Generating report for '{council}' ({year})...")
report = build_report(council, year)
pdf = render_pdf(report)

slug = report.council.local_authority_name.lower().replace(" ", "_")
out = Path(__file__).parent / f"{slug}_{year}.pdf"
out.write_bytes(pdf)
print(f"Written: {out}")
