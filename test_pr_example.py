"""Example file for testing OpenCodeReview PR review."""


def calculate_discount(price: float, percent: float) -> float:
    """Calculate discounted price.

    Args:
        price: Original price (must be non-negative).
        percent: Discount percentage (must be between 0 and 100).

    Returns:
        The discounted price.

    Raises:
        ValueError: If price is negative or percent is outside 0-100 range.
    """
    if price < 0:
        raise ValueError("Price must be non-negative")
    if percent < 0 or percent > 100:
        raise ValueError("Percent must be between 0 and 100")
    return price * (1 - percent / 100)


def format_username(first: str, last: str) -> str:
    """Format a username from first and last name."""
    return f"{first.lower()}.{last.lower()}"
