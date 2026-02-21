"""Base adapter interface for converting a ParsedIntent into API payloads.

Each external API (IBex, EPC register, flood-risk service, etc.) gets its
own adapter subclass that knows how to translate our canonical intent into
the request format that API expects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..schema import ParsedIntent


class BaseAdapter(ABC):
    """Abstract base class for API adapters.

    Subclasses translate a ``ParsedIntent`` into one or more API-specific
    request payloads and know how to call the API and normalise the response.
    """

    @abstractmethod
    def build_payloads(self, intent: ParsedIntent) -> list[dict[str, Any]]:
        """Convert a ParsedIntent into API request payload(s).

        Args:
            intent: The canonical parsed intent.

        Returns:
            A list of request-body dicts ready to send to the API.
        """
        ...

    @abstractmethod
    async def execute(self, intent: ParsedIntent) -> list[dict[str, Any]]:
        """Build payloads, call the API, and return normalised results.

        Args:
            intent: The canonical parsed intent.

        Returns:
            A list of normalised result dicts.
        """
        ...
