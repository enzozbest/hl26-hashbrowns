"""Abstract base class for report sections.

Each section receives the resolved CouncilContext and any pre-fetched
DataFrames it declared as dependencies, then returns a SectionResult.

To add a new section:
    1. Subclass BaseSection.
    2. Set ``section_id`` and ``title``.
    3. Implement ``run()``.
    4. Add an instance to ``report/sections/__init__.py::SECTIONS``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from report.models import CouncilContext, SectionResult


class BaseSection(ABC):
    """Contract that every report section must satisfy."""

    #: Stable snake_case identifier — used to reference sections by ID.
    section_id: str

    #: Display heading shown in the rendered PDF.
    title: str

    @abstractmethod
    def run(self, council: CouncilContext, data: dict[str, pd.DataFrame]) -> SectionResult:
        """Compute the section result.

        Args:
            council:  Resolved council identity.
            data:     Pre-fetched DataFrames keyed by table name.
                      Sections should declare which tables they need
                      via ``required_tables`` so the builder can load them.

        Returns:
            A fully populated :class:`~report.models.SectionResult`.
        """

    @property
    def required_tables(self) -> list[str]:
        """DB table names this section needs.  Override to declare dependencies."""
        return []
