"""LLM-based intent parser.

Takes a raw natural language query and returns a ParsedIntent by prompting
an LLM with structured output instructions.

TODO: implement — this is the scaffold.
"""

from __future__ import annotations

from .schema import ParsedIntent


async def parse_query(query: str) -> ParsedIntent:
    """Parse a natural language planning query into a structured ParsedIntent.

    Args:
        query: The raw user input, e.g. "I want to build 20 affordable flats
               in South London, avoiding flood zones".

    Returns:
        A fully populated ``ParsedIntent``.
    """
    raise NotImplementedError("LLM parser not yet implemented")
