from typing import Any, Dict


# Approximate pricing per 1M tokens.
# Keep these configurable/updateable as model pricing changes.
MODEL_PRICING = {
    "gpt-4.1-mini": {
        "input": 0.40,
        "output": 1.60,
    },
        "nex-agi/nex-n2.5-mini:free": {
        "input": 0.60,
        "output": 1.20,
    },
}


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """
    Estimate LLM cost in USD.

    Pricing is stored as USD per 1M tokens.
    """

    pricing = MODEL_PRICING.get(model)

    if not pricing:
        return 0.0

    input_cost = (
        input_tokens
        / 1_000_000
        * pricing["input"]
    )

    output_cost = (
        output_tokens
        / 1_000_000
        * pricing["output"]
    )

    return round(
        input_cost + output_cost,
        8,
    )


def extract_usage(
    response: Any,
) -> Dict[str, int]:
    """Extract token usage from an LLM response."""

    usage = getattr(
        response,
        "usage_metadata",
        None,
    )

    if usage:
        return {
            "input_tokens": int(
                usage.get(
                    "input_tokens",
                    0,
                )
            ),
            "output_tokens": int(
                usage.get(
                    "output_tokens",
                    0,
                )
            ),
            "total_tokens": int(
                usage.get(
                    "total_tokens",
                    0,
                )
            ),
        }

    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }