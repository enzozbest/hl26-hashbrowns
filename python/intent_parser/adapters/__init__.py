"""Data source adapters — translate ParsedIntent into API-specific queries."""

from .base import DataSourceAdapter
from .constraints import ConstraintsAdapter
from .epc import EPCAdapter
from .flood import FloodRiskAdapter
from .ibex import IBexAdapter
from .price_paid import PricePaidAdapter

__all__ = [
    "ConstraintsAdapter",
    "DataSourceAdapter",
    "EPCAdapter",
    "FloodRiskAdapter",
    "IBexAdapter",
    "PricePaidAdapter",
]
