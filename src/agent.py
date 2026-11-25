"""Conversational state machine for CEO AI.

The agent asks one question at a time, parses answers in English, French and
Tunisian Arabic (Arabic or Latin script), fills sensible market defaults on
"skip", calls the ML scoring engine, and answers follow-up questions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .config import get_bot_name, get_preset, sector_defaults, sector_label, sector_names
from .i18n import t
from .nlg import answer_followup, report_summary_markdown
from .scoring import evaluate_idea
from .storage import save_evaluation
from .utils import normalize_language, parse_number


@dataclass
class AgentState:
    mode: str = "idea"
    language: str = "en"
    preset_name: str = "tunisia"
    description: str = ""
    fields: dict[str, Any] = field(default_factory=dict)
    field_index: int = 0
    stage: str = "description"
    ready: bool = False
    result: dict[str, Any] | None = None
    default_used: list[str] = field(default_factory=list)

    @property
    def collected_count(self) -> int:
        return len(self.fields) + (1 if self.description else 0)

    def reset(self) -> None:
        self.description = ""
        self.fields = {}
        self.field_index = 0
        self.stage = "description"
        self.ready = False
        self.result = None
        self.default_used = []


@dataclass
class AgentTurn:
    response: str
    ready: bool = False
    result: dict[str, Any] | None = None
    current_field: str | None = None
    default_used: str | None = None


FIELD_ORDER: list[str] = [
    "sector",
    "target_market",
    "location",
    "budget",
    "team_size",
    "price",
    "expected_customers",
    "revenue_model",
    "competitor_intensity",
    "differentiation",
    "timeline_months",
]

_QUESTION_KEYS = {name: f"ask_{name}" for name in FIELD_ORDER}
_QUESTION_KEYS["target_market"] = "ask_target"
_QUESTION_KEYS["timeline_months"] = "ask_timeline"
_QUESTION_KEYS["competitor_intensity"] = "ask_competition"
_QUESTION_KEYS["expected_customers"] = "ask_customers"
_QUESTION_KEYS["revenue_model"] = "ask_revenue_model"

_SKIP_WORDS = (
    "skip",
    "pass",
    "default",
    "i don't know",
    "dont know",
    "je ne sais pas",
    "passer",
    "défaut",
    "defaut",
    "تجاوز",
    "تخطى",
    "ما نعرفش",
    "ماعرفش",
    "افتراضي",
)
_EVALUATE_NOW = (
    "evaluate now",
    "evaluate",
    "score now",
    "run now",
    "évaluer maintenant",
    "evaluer maintenant",
    "évalue maintenant",
    "evaluer",
    "résultat",
    "resultat",
    "قيّم الآن",
    "قيم الان",
    "قيّم",
    "احسب",
    "النتيجة",
)
_RESET_WORDS = (
    "new idea",
    "new conversation",
    "start over",
    "reset",
    "recommencer",
    "nouvelle idée",
    "nouvelle idee",
    "réinitialiser",
    "reinitialiser",
    "فكرة جديدة",
    "محادثة جديدة",
    "إعادة",
    "اعادة",
)

_SECTOR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "food_beverage": (
        "food",
        "restaurant",
        "cafe",
        "café",
        "bakery",
        "cuisine",
        "aliment",
        "restauration",
        "مطعم",
        "قهوة",
        "أكل",
        "طعام",
    ),
    "retail": ("retail", "shop", "store", "boutique", "commerce", "magasin", "متجر", "تجارة"),
    "ecommerce": ("ecommerce", "e-commerce", "online store", "dropshipping", "marketplace online", "بيع أونلاين", "تجارة إلكترونية"),
    "saas": ("saas", "software", "app", "application", "platform", "tech", "logiciel", "تطبيق", "برنامج", "منصة"),
    "fintech": ("fintech", "finance", "payment", "bank", "insurance", "paiement", "banque", "مالية", "دفع", "بنك"),
    "education": ("education", "school", "course", "training", "learning", "école", "ecole", "formation", "تعليم", "تدريب", "دورة"),
    "healthcare": ("health", "clinic", "medical", "healthcare", "santé", "sante", "clinique", "صحة", "عيادة", "طبي"),
    "logistics": ("logistics", "delivery", "transport", "shipping", "livraison", "logistique", "لوجستيك", "توصيل", "نقل"),
    "tourism": ("tourism", "hotel", "travel", "tourisme", "hôtel", "hotel", "voyage", "سياحة", "فندق", "سفر"),
    "renewable_energy": ("energy", "solar", "renewable", "green", "énergie", "energie", "solaire", "طاقة", "شمسية", "متجددة"),
    "creative_services": ("creative", "design", "marketing agency", "consulting", "agency", "créatif", "creation", "استوديو", "تصميم", "استشارات"),
    "manufacturing": ("manufacturing", "factory", "production", "industry", "usine", "fabrication", "industrie", "صناعة", "مصنع"),
    "agriculture": ("agriculture", "farm", "agri", "agricole", "ferme", "فلاحة", "زراعة", "مزرعة"),
    "real_estate": ("real estate", "property", "immobilier", "proptech", "عقار", "عقاري", "إسكان"),
    "fitness_wellness": ("fitness", "gym", "wellness", "sport", "yoga", "salle de sport", "لياقة", "رياضة", "جيم"),
}

_REVENUE_KEYWORDS = {
    "subscription": ("subscription", "abonnement", "abonnement mensuel", "اشتراك", "شهري"),
    "one_time": ("one time", "one-time", "one shot", "purchase", "achat", "fois", "مرة واحدة", "بيع"),
    "commission": ("commission", "take rate", "عمولة"),
    "freemium": ("freemium", "free", "gratuit", "مجاني"),
    "advertising": ("advertising", "ads", "publicité", "publicite", "إعلانات", "اعلانات"),
    "hybrid": ("hybrid", "mix", "hybride", "مختلط", "مزيج"),
}

_LOW_WORDS = ("low", "weak", "faible", "peu", "easy", "facile", "ضعيف", "قليل", "سهل", "خفيف")
_MEDIUM_WORDS = ("medium", "moyen", "moyenne", "moderate", "متوسط", "عادي")
_HIGH_WORDS = ("high", "strong", "fort", "élevé", "eleve", "intense", "saturated", "قوي", "شديد", "مرتفع", "كثيف")
_UNIQUE_WORDS = ("unique", "different", "differentiated", "innovative", "mieux", "différent", "متميز", "فريد", "مختلف", "جديد")
_COPY_WORDS = ("same", "copy", "copie", "identique", "commodity", "تقليد", "نفس", "عادي")


def _has_any(text: str, words: tuple[str, ...]) -> bool:
    return any(word in text for word in words)


def _clean(text: str) -> str:
    text = text.strip().lower().replace("’", "'")
    return re.sub(r"\s+", " ", text)


def _parse_sector(text: str) -> str | None:
    cleaned = _clean(text)
    # E-commerce and food are checked before generic retail/food words.
    priority = [
        "ecommerce",
        "food_beverage",
        "saas",
        "fintech",
        "healthcare",
        "renewable_energy",
        "real_estate",
        "logistics",
        "education",
        "tourism",
        "fitness_wellness",
        "creative_services",
        "manufacturing",
        "agriculture",
        "retail",
    ]
    for sector in priority:
        if _has_any(cleaned, _SECTOR_KEYWORDS[sector]):
            return sector
    for sector in sector_names():
        if sector.replace("_", " ") in cleaned:
            return sector
    return None


def _parse_target_market(text: str) -> str:
    cleaned = _clean(text)
    if "b2b2c" in cleaned or "b2b2c" in cleaned.replace(" ", ""):
        return "b2b2c"
    if _has_any(cleaned, ("marketplace", "market place", "platform", "سوق", "منصة")):
        return "marketplace"
    if _has_any(cleaned, ("b2b", "business", "companies", "company", "entreprise", "entreprises", "شركات", "مؤسسات")):
        return "b2b"
    return "b2c"


def _parse_revenue_model(text: str) -> str | None:
    cleaned = _clean(text)
    for model, words in _REVENUE_KEYWORDS.items():
        if _has_any(cleaned, words):
            return model
    return None


def _parse_rating(text: str, *, invert: bool = False) -> float | None:
    cleaned = _clean(text)
    value = parse_number(cleaned)
    if value is not None:
        if "%" in cleaned:
            value = value / 10
        return max(1.0, min(10.0, float(value)))
    low = any(word in cleaned for word in _LOW_WORDS)
    high = any(word in cleaned for word in _HIGH_WORDS)
    unique = any(word in cleaned for word in _UNIQUE_WORDS)
    copy = any(word in cleaned for word in _COPY_WORDS)
    if invert:
        if unique:
            return 8.5
        if copy:
            return 3.5
    if low:
        return 3.0
    if high:
        return 9.0
    if any(word in cleaned for word in _MEDIUM_WORDS):
        return 6.0
    return None


def _parse_timeline(text: str) -> float | None:
    cleaned = _clean(text)
    value = parse_number(cleaned)
    if value is None:
        if _has_any(cleaned, ("year", "an", "année", "annee", "سنة")):
            return 12
        if _has_any(cleaned, ("week", "semaine", "أسبوع", "اسبوع")):
            return 1
        return None
    if _has_any(cleaned, ("year", "an", "année", "annee", "سنة")):
        value *= 12
    elif _has_any(cleaned, ("week", "semaine", "أسبوع", "اسبوع")):
        value = max(1, value / 4.33)
    return float(max(1, round(value)))


def _parse_positive_number(text: str) -> float | None:
    value = parse_number(text)
    if value is None or value <= 0:
        return None
    return float(value)


def _location_default(preset_name: str, language: str) -> str:
    preset = get_preset(preset_name)
    cities = preset.get("cities", {})
    values = cities.get(language) or cities.get("en") or ["Capital city"]
    return str(values[0])


def _defaults(state: AgentState) -> dict[str, Any]:
    sector = str(state.fields.get("sector", "other"))
    defaults = sector_defaults(sector, state.preset_name)
    defaults["sector"] = sector
    defaults["location"] = _location_default(state.preset_name, state.language)
    defaults["target_market"] = "b2c"
    defaults["revenue_model"] = "one_time"
    return defaults


def _question(state: AgentState, field_name: str) -> str:
    defaults = _defaults(state)
    if field_name == "sector":
        options = ", ".join(sector_label(sector, state.language) for sector in sector_names())
        return t("ask_sector", state.language, options=options)
    if field_name == "location":
        return f"{t('ask_location', state.language)} ({_location_default(state.preset_name, state.language)})"
    if field_name == "budget":
        return f"{t('ask_budget', state.language)} [{defaults['budget']:,.0f}]"
    if field_name == "team_size":
        return f"{t('ask_team', state.language)} [{defaults['team_size']:.0f}]"
    if field_name == "price":
        return f"{t('ask_price', state.language)} [{defaults['price']:,.0f}]"
    if field_name == "expected_customers":
        return f"{t('ask_customers', state.language)} [{defaults['expected_customers']:,.0f}]"
    if field_name == "competitor_intensity":
        return f"{t('ask_competition', state.language)} [{defaults['competitor_intensity']:.0f}]"
    if field_name == "differentiation":
        return f"{t('ask_differentiation', state.language)} [{defaults['differentiation']:.0f}]"
    if field_name == "timeline_months":
        return f"{t('ask_timeline', state.language)} [{defaults['timeline_months']:.0f}]"
    return t(_QUESTION_KEYS.get(field_name, "ask_description"), state.language)


def _extract_value(state: AgentState, field_name: str, text: str) -> Any:
    defaults = _defaults(state)
    cleaned = _clean(text)
    if field_name == "sector":
        return _parse_sector(cleaned) or defaults["sector"]
    if field_name == "target_market":
        return _parse_target_market(cleaned)
    if field_name == "location":
        return text.strip() or defaults["location"]
    if field_name in {"budget", "price", "expected_customers", "team_size"}:
        value = _parse_positive_number(cleaned)
        if field_name == "team_size" and value is not None:
            value = max(1, round(value))
        return value if value is not None else defaults[field_name]
    if field_name == "revenue_model":
        return _parse_revenue_model(cleaned) or defaults["revenue_model"]
    if field_name == "competitor_intensity":
        value = _parse_rating(cleaned)
        return value if value is not None else defaults["competitor_intensity"]
    if field_name == "differentiation":
        value = _parse_rating(cleaned, invert=True)
        return value if value is not None else defaults["differentiation"]
    if field_name == "timeline_months":
        value = _parse_timeline(cleaned)
        return value if value is not None else defaults["timeline_months"]
    return text.strip()


def _next_question(state: AgentState) -> str:
    if state.field_index >= len(FIELD_ORDER):
        return ""
    return _question(state, FIELD_ORDER[state.field_index])


def _fill_remaining_defaults(state: AgentState) -> None:
    defaults = _defaults(state)
    for field_name in FIELD_ORDER:
        if field_name not in state.fields:
            state.fields[field_name] = defaults[field_name]


def _run_evaluation(state: AgentState) -> dict[str, Any]:
    if not state.description:
        raise ValueError("A description is required before evaluation.")
    _fill_remaining_defaults(state)
    inputs = {
        "description": state.description,
        **state.fields,
        "language": state.language,
    }
    result = evaluate_idea(inputs, state.preset_name, state.language)
    result["created_inputs"] = inputs
    try:
        result["report_id"] = save_evaluation(result)
    except Exception:
        result["report_id"] = None
    state.result = result
    state.ready = True
    state.stage = "complete"
    return result


def start_conversation(language: str = "en", preset_name: str = "tunisia", mode: str = "idea") -> tuple[AgentState, str]:
    state = AgentState(
        mode=mode,
        language=normalize_language(language),
        preset_name=preset_name,
    )
    if mode == "business":
        return state, t("business_greeting", state.language)
    greeting = t("agent_greeting", state.language, bot=get_bot_name())
    return state, f"{greeting}\n\n{t('evaluation_now', state.language)}"


def process_message(state: AgentState, message: str) -> AgentTurn:
    """Advance the state machine by one user message."""
    cleaned = _clean(message)
    if not cleaned:
        return AgentTurn(response=_question(state, FIELD_ORDER[state.field_index]) if state.stage != "complete" else t("answer_default", state.language))

    if _has_any(cleaned, _RESET_WORDS):
        state.reset()
        return AgentTurn(
            response=(
                f"{t('greeting_short', state.language, bot=get_bot_name())}\n\n"
                f"{t('ask_description', state.language)}"
            ),
            current_field="description",
        )

    if state.ready and state.result is not None:
        return AgentTurn(
            response=answer_followup(state.result, message, state.language),
            ready=True,
            result=state.result,
        )

    if state.stage == "description":
        if _has_any(cleaned, _EVALUATE_NOW):
            return AgentTurn(response=t("ask_description", state.language), current_field="description")
        state.description = message.strip()
        state.stage = "fields"
        state.field_index = 0
        return AgentTurn(
            response=f"{t('received', state.language)}\n\n{_question(state, FIELD_ORDER[0])}",
            current_field=FIELD_ORDER[0],
        )

    if _has_any(cleaned, _EVALUATE_NOW):
        result = _run_evaluation(state)
        return AgentTurn(
            response=f"{t('evaluation_started', state.language)}\n\n{report_summary_markdown(result, state.language)}\n\n{t('ask_anything', state.language)}",
            ready=True,
            result=result,
        )

    current = FIELD_ORDER[state.field_index]
    was_skip = _has_any(cleaned, _SKIP_WORDS)
    default_used = None
    if was_skip:
        value = _defaults(state)[current]
        default_used = str(value)
        state.default_used.append(current)
    else:
        value = _extract_value(state, current, message)
        # Detect whether the parser had to fall back to the default.
        defaults = _defaults(state)
        if str(value) == str(defaults.get(current)):
            default_used = str(value)
    state.fields[current] = value
    state.field_index += 1

    ack = t("received", state.language)
    if default_used is not None:
        ack += " " + t("default_used", state.language, value=default_used)

    if state.field_index >= len(FIELD_ORDER):
        result = _run_evaluation(state)
        return AgentTurn(
            response=(
                f"{ack}\n\n{t('evaluation_started', state.language)}\n\n"
                f"{report_summary_markdown(result, state.language)}\n\n{t('ask_anything', state.language)}"
            ),
            ready=True,
            result=result,
            default_used=default_used,
        )
    next_field = FIELD_ORDER[state.field_index]
    return AgentTurn(
        response=f"{ack}\n\n{_question(state, next_field)}",
        current_field=next_field,
        default_used=default_used,
    )


def update_preferences(state: AgentState, language: str | None = None, preset_name: str | None = None) -> None:
    if language:
        state.language = normalize_language(language)
    if preset_name:
        state.preset_name = preset_name
