"""Small shared helpers (numbers, formatting, language normalization)."""

from __future__ import annotations

import json
import math
import re
from datetime import date, datetime
from typing import Any


def clamp(value: float, low: float, high: float) -> float:
    """Clamp *value* between *low* and *high*."""
    return max(low, min(high, value))


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float without raising on missing/invalid data."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        try:
            if math.isnan(float(value)) or math.isinf(float(value)):
                return default
        except (TypeError, ValueError):
            return default
        return float(value)
    text = str(value).strip().replace(",", ".")
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def parse_number(text: str | None, default: float | None = None) -> float | None:
    """Extract a human-friendly number from text.

    Understands ``30 000``, ``30,000``, ``30k``, ``2.5m`` and percentages.
    """
    if text is None:
        return default
    raw = str(text).strip().lower().replace("\u00a0", " ")
    if not raw:
        return default
    # Keep the first numeric expression, including common thousand separators.
    match = re.search(r"-?\d[\d\s,\.]*", raw)
    if not match:
        return default
    number_text = match.group(0).strip().replace(" ", "")
    multiplier = 1.0
    tail = raw[match.end() :].strip()
    if tail.startswith("k") or raw.endswith("k"):
        multiplier = 1_000
    elif tail.startswith("m") or raw.endswith("m"):
        multiplier = 1_000_000
    if "." in number_text and "," in number_text:
        number_text = number_text.replace(",", "")
    else:
        number_text = number_text.replace(",", ".")
    # Remove duplicated decimal dots caused by odd input.
    if number_text.count(".") > 1:
        parts = number_text.split(".")
        number_text = "".join(parts[:-1]) + "." + parts[-1]
    try:
        return float(number_text) * multiplier
    except ValueError:
        return default


def normalize_language(language: str | None) -> str:
    """Return one of the three supported language codes."""
    code = (language or "en").strip().lower()
    aliases = {
        "english": "en",
        "anglais": "en",
        "fr": "fr",
        "français": "fr",
        "francais": "fr",
        "french": "fr",
        "ar": "ar",
        "arabic": "ar",
        "arabe": "ar",
        "tn": "ar",
        "تونسي": "ar",
    }
    return aliases.get(code, code if code in {"en", "fr", "ar"} else "en")


def format_money(value: float | None, currency: str = "TND", lang: str = "en") -> str:
    """Format a monetary amount with a compact but readable style."""
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if abs(value) >= 1_000_000:
        compact = f"{value / 1_000_000:,.1f}M"
    elif abs(value) >= 10_000:
        compact = f"{value / 1_000:,.0f}k"
    else:
        compact = f"{value:,.0f}"
    return f"{compact} {currency}"


def format_pct(value: float | None, digits: int = 1) -> str:
    """Format a value that is already expressed in percent."""
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}%"


def json_safe(value: Any) -> Any:
    """Recursively convert numpy/pandas/date values into JSON-safe objects."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def dumps_safe(value: Any) -> str:
    return json.dumps(json_safe(value), ensure_ascii=False)


def title_case_key(key: str) -> str:
    return key.replace("_", " ").strip().title()
