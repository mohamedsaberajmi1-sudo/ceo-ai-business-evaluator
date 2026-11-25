"""Configuration loader for CEO AI.

The bot name, paths, LLM settings and scoring weights are read from
``configs/app.yaml``. Market presets and sector assumptions live in
``configs/markets.yaml`` so they can be edited without touching Python.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .utils import normalize_language

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

_SUPPORTED_LANGUAGES = ("en", "fr", "ar")

# The exact footer sentence is intentionally kept here, next to the single
# BOT_NAME source, and reused by the UI and the PDF generator.
_FOOTER_TEMPLATES = {
    "en": "{bot_name} is AI and can make mistakes.",
    "fr": "{bot_name} est une IA et peut faire des erreurs.",
    "ar": "{bot_name} ذكاء اصطناعي وقد يرتكب أخطاء.",
}


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    temperature: float
    timeout_seconds: int
    ollama_base_url: str
    openai_key: str
    anthropic_key: str

    @property
    def enabled(self) -> bool:
        return self.provider in {"ollama", "openai", "anthropic"}


@dataclass(frozen=True)
class AppPaths:
    data: Path
    models: Path
    evaluations_db: Path
    markets_config: Path


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return data


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """Load and cache ``configs/app.yaml``."""
    return _read_yaml(ROOT_DIR / "configs" / "app.yaml")


@lru_cache(maxsize=1)
def get_markets() -> dict[str, Any]:
    """Load and cache ``configs/markets.yaml``."""
    config = get_config()
    relative = config.get("paths", {}).get("markets_config", "configs/markets.yaml")
    return _read_yaml(ROOT_DIR / relative)


def get_bot_name() -> str:
    return str(get_config().get("bot_name", "CEO AI"))


def get_paths() -> AppPaths:
    config = get_config()
    paths = config.get("paths", {})
    db_override = os.getenv("EVALUATIONS_DB")
    return AppPaths(
        data=ROOT_DIR / paths.get("data", "data"),
        models=ROOT_DIR / paths.get("models", "models"),
        evaluations_db=ROOT_DIR / (db_override or paths.get("evaluations_db", "data/evaluations.db")),
        markets_config=ROOT_DIR / paths.get("markets_config", "configs/markets.yaml"),
    )


def ensure_directories() -> None:
    paths = get_paths()
    for directory in (paths.data, paths.models, paths.evaluations_db.parent):
        directory.mkdir(parents=True, exist_ok=True)


def get_llm_settings() -> LLMSettings:
    config = get_config().get("llm", {})
    provider = os.getenv("LLM_PROVIDER", str(config.get("provider", "none"))).strip().lower()
    if provider not in {"none", "ollama", "openai", "anthropic"}:
        provider = "none"
    return LLMSettings(
        provider=provider,
        model=os.getenv("LLM_MODEL", str(config.get("model", ""))).strip(),
        temperature=float(config.get("temperature", 0.2)),
        timeout_seconds=int(config.get("timeout_seconds", 20)),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        openai_key=os.getenv("OPENAI_API_KEY", "").strip(),
        anthropic_key=os.getenv("ANTHROPIC_API_KEY", "").strip(),
    )


def get_language_name(language: str) -> str:
    languages = get_config().get("supported_languages", {})
    return str(languages.get(normalize_language(language), "English"))


def footer_text(language: str = "en") -> str:
    """Return the required localized footer sentence, with no link."""
    language = normalize_language(language)
    template = _FOOTER_TEMPLATES.get(language, _FOOTER_TEMPLATES["en"])
    return template.format(bot_name=get_bot_name())


def get_preset(name: str) -> dict[str, Any]:
    presets = get_markets().get("presets", {})
    if name not in presets:
        raise KeyError(f"Unknown market preset: {name}")
    return presets[name]


def preset_names() -> list[str]:
    return list(get_markets().get("presets", {}).keys())


def preset_label(name: str, language: str = "en") -> str:
    preset = get_preset(name)
    labels = preset.get("label", {})
    return str(labels.get(normalize_language(language), labels.get("en", name)))


def get_sector_profile(sector: str) -> dict[str, Any]:
    sectors = get_markets().get("sectors", {})
    if sector not in sectors:
        sector = "other"
    return dict(sectors[sector])


def sector_names() -> list[str]:
    return list(get_markets().get("sectors", {}).keys())


def sector_label(sector: str, language: str = "en") -> str:
    profile = get_market_sector_raw(sector)
    labels = profile.get("label", {})
    return str(labels.get(normalize_language(language), labels.get("en", sector)))


def get_market_sector_raw(sector: str) -> dict[str, Any]:
    sectors = get_markets().get("sectors", {})
    return dict(sectors.get(sector, sectors.get("other", {})))


def sector_defaults(sector: str, preset_name: str) -> dict[str, float]:
    """Reasonable defaults for a sector/preset combination."""
    profile = get_sector_profile(sector)
    preset = get_preset(preset_name)
    return {
        "budget": float(preset.get("typical_budget", 30000)),
        "price": float(profile.get("typical_price", preset.get("typical_price", 50))),
        "expected_customers": float(profile.get("typical_customers", preset.get("typical_customers", 150))),
        "team_size": float(preset.get("typical_team_size", 3)),
        "marketing_spend": float(preset.get("typical_marketing", 500)),
        "timeline_months": float(preset.get("typical_timeline_months", 9)),
        "competitor_intensity": float(profile.get("competition", 7)),
        "differentiation": 6.0,
    }


def currency_for(preset_name: str) -> str:
    return str(get_preset(preset_name).get("currency_symbol", "TND"))


def supported_languages() -> tuple[str, ...]:
    return _SUPPORTED_LANGUAGES
