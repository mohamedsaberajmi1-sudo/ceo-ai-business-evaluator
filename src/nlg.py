"""Natural-language generation: offline templates + optional pluggable LLM."""

from __future__ import annotations

import json
from typing import Any

import requests

from .config import get_llm_settings
from .i18n import t
from .utils import format_money

_LANG_NAMES = {"en": "English", "fr": "French", "ar": "Tunisian Arabic (Arabic script)"}


def _money(report: dict[str, Any], value: float | None) -> str:
    return format_money(value, report.get("currency", "TND"))


def _build_context(report: dict[str, Any]) -> str:
    compact = {
        "mode": report.get("mode"),
        "score": report.get("overall_score"),
        "success_probability": report.get("success_probability"),
        "confidence": report.get("confidence"),
        "verdict": report.get("verdict"),
        "subscores": report.get("subscores"),
        "financials": report.get("financials"),
        "key_risks": report.get("key_risks"),
        "modifications": report.get("modifications"),
        "swot": report.get("swot"),
        "kpis": report.get("kpis"),
        "problems": report.get("problems"),
        "actions": report.get("actions"),
        "benchmarks": report.get("benchmarks"),
        "forecast_method": (report.get("forecast") or {}).get("method"),
        "disclaimer": report.get("disclaimer"),
    }
    return json.dumps(compact, ensure_ascii=False, default=str)[:9000]


def llm_generate(system: str, prompt: str, language: str = "en") -> str | None:
    """Call the configured LLM if available. Never raises."""
    settings = get_llm_settings()
    if not settings.enabled:
        return None
    try:
        if settings.provider == "ollama":
            response = requests.post(
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": settings.model or "llama3.1:8b",
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "options": {"temperature": settings.temperature},
                },
                timeout=settings.timeout_seconds,
            )
            response.raise_for_status()
            return str(response.json().get("message", {}).get("content", "")).strip() or None
        if settings.provider == "openai" and settings.openai_key:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.openai_key}"},
                json={
                    "model": settings.model or "gpt-4o-mini",
                    "temperature": settings.temperature,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=settings.timeout_seconds,
            )
            response.raise_for_status()
            return str(response.json()["choices"][0]["message"]["content"]).strip() or None
        if settings.provider == "anthropic" and settings.anthropic_key:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": settings.anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": settings.model or "claude-3-5-haiku-latest",
                    "max_tokens": 900,
                    "temperature": settings.temperature,
                    "system": system,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=settings.timeout_seconds,
            )
            response.raise_for_status()
            blocks = response.json().get("content", [])
            return "\n".join(block.get("text", "") for block in blocks if block.get("type") == "text").strip() or None
    except Exception:
        return None
    return None


def llm_status() -> str:
    settings = get_llm_settings()
    if not settings.enabled:
        return "offline"
    if settings.provider == "ollama":
        return "online"
    if settings.provider == "openai" and settings.openai_key:
        return "online"
    if settings.provider == "anthropic" and settings.anthropic_key:
        return "online"
    return "error"


def _system_prompt(language: str) -> str:
    return (
        f"You are a concise business-analysis assistant. Answer in {_LANG_NAMES.get(language, 'English')}. "
        "Use only the supplied report context, never invent numbers. Explain estimates clearly. "
        "Always mention that ROI and profit are estimates, not financial advice. "
        "Do not output links or Markdown tables; use short paragraphs and bullet points."
    )


def report_summary_markdown(report: dict[str, Any], language: str | None = None) -> str:
    """Create the offline chat summary shown right after an evaluation."""
    language = language or report.get("language", "en")
    if report.get("mode") == "business":
        kpis = report.get("kpis", {})
        return "\n".join(
            [
                f"**{t('analysis_complete', language)}**",
                "",
                report.get("summary", ""),
                "",
                f"- **{t('profit', language)}:** {_money(report, kpis.get('last_month_profit'))}",
                f"- **{t('margin', language)}:** {kpis.get('net_margin_pct', 0):.1f}%",
                f"- **{t('growth', language)}:** {kpis.get('monthly_growth_pct', 0):.1f}%",
                f"- **{t('churn', language)}:** {kpis.get('churn_pct', 0):.1f}%",
                f"- **{t('confidence', language)}:** {report.get('confidence', 0):.0f}%",
                "",
                f"> {report.get('disclaimer', '')}",
            ]
        )
    financials = report.get("financials", {})
    return "\n".join(
        [
            f"**{t('report_ready', language)}**",
            "",
            f"**{t('verdict', language)}:** {report.get('verdict', '')}",
            f"**{t('overall_score', language)}:** {report.get('overall_score', 0):.1f}/100  ·  "
            f"**{t('success_probability', language)}:** {report.get('success_probability', 0):.1f}%  ·  "
            f"**{t('confidence', language)}:** {report.get('confidence', 0):.0f}%",
            f"**{t('roi', language)}:** {financials.get('roi_percent', 0):.1f}%  ·  "
            f"**{t('breakeven', language)}:** {financials.get('breakeven_months', 0):.0f} {t('months', language)}",
            "",
            f"- **{t('top_drivers', language)}:** {', '.join(report.get('positive_drivers', [])[:3])}",
            f"- **{t('key_risks', language)}:** {report.get('key_risks', [''])[0]}",
            f"- **{t('recommended_modifications', language)}:** {report.get('modifications', [{}])[0].get('action', '') if report.get('modifications') else ''}",
            "",
            f"> {report.get('disclaimer', '')}",
        ]
    )


def _keywords(text: str, words: tuple[str, ...]) -> bool:
    return any(word in text for word in words)


def _rule_based_answer(report: dict[str, Any], question: str, language: str) -> str:
    text = question.lower().strip()
    if report.get("mode") == "business":
        kpis = report.get("kpis", {})
        if _keywords(text, ("forecast", "prévision", "prevision", "توق", "future", "avenir")):
            method = (report.get("forecast") or {}).get("method", "")
            return (
                f"**{t('forecast', language)}** — {method}. "
                f"{t('business_insight', language)}"
            )
        if _keywords(text, ("problem", "risque", "risk", "مشكل", "خطر", "diagnostic")):
            problems = report.get("problems", [])
            if not problems:
                return t("no_problems", language)
            return "\n".join(f"- **{item['severity'].upper()}** — {item['title']}: {item['detail']}" for item in problems)
        if _keywords(text, ("improve", "action", "amélior", "amelior", "حسّن", "تحسين", "plan")):
            actions = report.get("actions", [])
            lines = [
                f"- **{item['action']}** — {t('estimated_impact', language)}: "
                f"{_money(report, item['impact_monthly'])}/{t('month', language)}"
                for item in actions[:4]
            ]
            return "\n".join(lines)
        if _keywords(text, ("kpi", "margin", "marge", "profit", "ربح", "chiffre", "ca", "إيراد")):
            return (
                f"**{t('kpis', language)}** — {t('revenue', language)}: {_money(report, kpis.get('last_month_revenue'))}; "
                f"{t('profit', language)}: {_money(report, kpis.get('last_month_profit'))}; "
                f"{t('margin', language)}: {kpis.get('net_margin_pct', 0):.1f}%; "
                f"{t('growth', language)}: {kpis.get('monthly_growth_pct', 0):.1f}%."
            )
        return report.get("summary", t("answer_default", language))

    # Mode A intents
    if _keywords(text, ("why", "pourquoi", "لماذا", "explain", "explique", "score")):
        return t(
            "answer_why",
            language,
            score=f"{report.get('overall_score', 0):.0f}",
            prob=f"{report.get('success_probability', 0):.0f}",
            positives=", ".join(report.get("positive_drivers", [])[:3]) or "—",
            negatives=", ".join(report.get("negative_drivers", [])[:3]) or "—",
        )
    if _keywords(text, ("roi", "profit", "rentab", "argent", "فلوس", "عائد", "ربح", "break", "seuil", "تعادل")):
        financials = report.get("financials", {})
        scenarios = report.get("scenarios", {})
        return t(
            "answer_roi",
            language,
            roi=f"{financials.get('roi_percent', 0):.1f}",
            months=f"{financials.get('breakeven_months', 0):.0f}",
            profit=_money(report, financials.get("monthly_profit")),
            low=_money(report, scenarios.get("pessimistic", {}).get("monthly_profit")),
            high=_money(report, scenarios.get("optimistic", {}).get("monthly_profit")),
        )
    if _keywords(text, ("risk", "risque", "danger", "خطر", "مخاطر", "fail")):
        risks = report.get("key_risks", [])[:3]
        actions = [item["action"] for item in report.get("modifications", [])[:2]]
        return t(
            "answer_risks",
            language,
            risks="; ".join(risks) or "—",
            actions="; ".join(actions) or "—",
        )
    if _keywords(text, ("improve", "amélior", "amelior", "conseil", "حسّن", "تحسين", "action", "plan")):
        modifications = report.get("modifications", [])
        actions = [item["action"] for item in modifications[:3]]
        best = modifications[0] if modifications else {"score_gain": 0, "roi_gain": 0}
        return t(
            "answer_improve",
            language,
            actions="; ".join(actions) or "—",
            points=f"{best.get('score_gain', 0):.1f}",
            roi_points=f"{best.get('roi_gain', 0):.0f}",
        )
    if _keywords(text, ("segment", "target", "cible", "جمهور", "عميل", "client")):
        return report.get("target_segment", "")
    if _keywords(text, ("summary", "résumé", "resume", "verdict", "خلاصة", "حكم")):
        financials = report.get("financials", {})
        return t(
            "answer_summary",
            language,
            verdict=report.get("verdict", ""),
            score=f"{report.get('overall_score', 0):.0f}",
            prob=f"{report.get('success_probability', 0):.0f}",
            roi=f"{financials.get('roi_percent', 0):.0f}",
            months=f"{financials.get('breakeven_months', 0):.0f}",
        )
    return t("answer_default", language)


def answer_followup(report: dict[str, Any], question: str, language: str = "en") -> str:
    """Answer a post-report question, optionally through the configured LLM."""
    fallback = _rule_based_answer(report, question, language)
    settings = get_llm_settings()
    if settings.enabled:
        prompt = (
            f"User question: {question}\n\n"
            f"Report context (JSON):\n{_build_context(report)}\n\n"
            "Answer in 3-6 short sentences using the report numbers. "
            "If the user asks for financial advice, restate that this is an estimate."
        )
        response = llm_generate(_system_prompt(language), prompt, language)
        if response:
            return response
    return fallback
