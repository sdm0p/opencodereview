"""Example file for testing OpenCodeReview PR review."""


def calculate_discount(price: float, percent: float) -> float:
    """Calculate discounted price."""
    return price * (1 - percent / 100)


def format_username(first: str, last: str) -> str:
    """Format a username from first and last name."""
    return f"{first.lower()}.{last.lower()}"
