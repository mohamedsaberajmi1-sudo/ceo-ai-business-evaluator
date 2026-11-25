"""PDF report generation with ReportLab, including the required localized footer."""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .config import footer_text, get_bot_name, get_preset, preset_label
from .i18n import t
from .utils import format_money, normalize_language

_FONT_NAME = "Helvetica"
_FONT_BOLD = "Helvetica-Bold"
_FONT_REGISTERED = False


def _register_font() -> None:
    global _FONT_NAME, _FONT_BOLD, _FONT_REGISTERED
    if _FONT_REGISTERED:
        return
    candidates = [
        ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    for regular, bold in candidates:
        try:
            pdfmetrics.registerFont(TTFont("AppFont", regular))
            pdfmetrics.registerFont(TTFont("AppFont-Bold", bold))
            _FONT_NAME = "AppFont"
            _FONT_BOLD = "AppFont-Bold"
            break
        except Exception:
            continue
    _FONT_REGISTERED = True


def _shape(text: str, language: str) -> str:
    if language != "ar":
        return text
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        return text


def _esc(text: Any, language: str) -> str:
    return _shape(escape(str(text)), language)


def _styles(language: str) -> dict[str, ParagraphStyle]:
    _register_font()
    base = getSampleStyleSheet()
    align = TA_RIGHT if language == "ar" else TA_LEFT
    return {
        "title": ParagraphStyle(
            "AppTitle",
            parent=base["Title"],
            fontName=_FONT_BOLD,
            fontSize=19,
            leading=24,
            textColor=colors.HexColor("#102A43"),
            alignment=align,
            spaceAfter=2 * mm,
        ),
        "subtitle": ParagraphStyle(
            "AppSubtitle",
            parent=base["Normal"],
            fontName=_FONT_NAME,
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#52606D"),
            alignment=align,
        ),
        "h2": ParagraphStyle(
            "AppH2",
            parent=base["Heading2"],
            fontName=_FONT_BOLD,
            fontSize=12.5,
            leading=16,
            textColor=colors.HexColor("#0B5FFF"),
            alignment=align,
            spaceBefore=5 * mm,
            spaceAfter=2 * mm,
        ),
        "body": ParagraphStyle(
            "AppBody",
            parent=base["BodyText"],
            fontName=_FONT_NAME,
            fontSize=8.8,
            leading=12.4,
            alignment=align,
            spaceAfter=1.2 * mm,
        ),
        "small": ParagraphStyle(
            "AppSmall",
            parent=base["BodyText"],
            fontName=_FONT_NAME,
            fontSize=7.2,
            leading=9.5,
            textColor=colors.HexColor("#6B7280"),
            alignment=align,
        ),
        "center": ParagraphStyle(
            "AppCenter",
            parent=base["Normal"],
            fontName=_FONT_NAME,
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
        ),
        "kpi": ParagraphStyle(
            "AppKpi",
            parent=base["Normal"],
            fontName=_FONT_BOLD,
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#0B5FFF"),
            alignment=TA_CENTER,
        ),
    }


def _fmt_money(value: Any, currency: str) -> str:
    try:
        return format_money(float(value), currency)
    except (TypeError, ValueError):
        return "—"


def _p(text: Any, style: ParagraphStyle, language: str) -> Paragraph:
    return Paragraph(_esc(text, language), style)


def _section(title: str, styles: dict[str, ParagraphStyle], language: str) -> list[Any]:
    return [Spacer(1, 1.5 * mm), _p(title, styles["h2"], language)]


def _bullet_list(items: list[str], styles: dict[str, ParagraphStyle], language: str) -> list[Any]:
    return [
        Paragraph("• " + _esc(item, language), styles["body"])
        for item in items
        if item
    ]


def _table(
    rows: list[list[Any]],
    styles: dict[str, ParagraphStyle],
    language: str,
    *,
    col_widths: list[float] | None = None,
    header: bool = True,
) -> Table:
    data: list[list[Any]] = []
    for row_index, row in enumerate(rows):
        data.append(
            [
                Paragraph(
                    _esc(value, language).replace("\n", "<br/>"),
                    styles["small"] if not (header and row_index == 0) else ParagraphStyle(
                        f"th{row_index}",
                        parent=styles["small"],
                        fontName=_FONT_BOLD,
                        textColor=colors.white,
                    ),
                )
                for value in row
            ]
        )
    table = Table(data, colWidths=col_widths, repeatRows=1 if header else 0, hAlign="LEFT")
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D9E2EC")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B5FFF")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ]
        style += [("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FA")])]
    else:
        style += [("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F5F7FA")])]
    table.setStyle(TableStyle(style))
    return table


def _score_chart(report: dict[str, Any], language: str) -> Image | None:
    criteria = report.get("criteria", [])
    if not criteria:
        return None
    labels = [item["label"] for item in criteria][::-1]
    scores = [float(item["score"]) for item in criteria][::-1]
    colors_map = ["#D64545" if score < 5 else "#F0B429" if score < 7 else "#2F855A" for score in scores]
    fig, ax = plt.subplots(figsize=(7.2, 3.55))
    ax.barh(labels, scores, color=colors_map, height=0.64)
    ax.set_xlim(0, 10)
    ax.set_xlabel("Score / 10")
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    ax.set_title(t("chart_scores", language), fontsize=10, loc="left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return Image(buffer, width=170 * mm, height=78 * mm)


def _forecast_chart(report: dict[str, Any], language: str) -> Image | None:
    records = (report.get("forecast") or {}).get("revenue") or []
    if not records:
        return None
    frame = pd.DataFrame(records)
    frame["date"] = pd.to_datetime(frame["date"])
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    ax.plot(frame["date"], frame["forecast"], color="#0B5FFF", marker="o", linewidth=1.8, label="Forecast")
    if {"lower", "upper"}.issubset(frame.columns):
        ax.fill_between(frame["date"], frame["lower"], frame["upper"], color="#0B5FFF", alpha=0.13, label="80% band")
    ax.set_title(t("chart_forecast", language), fontsize=10, loc="left")
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return Image(buffer, width=172 * mm, height=70 * mm)


def _on_page(canvas: Any, doc: Any, language: str) -> None:
    _register_font()
    canvas.saveState()
    width, height = A4
    canvas.setFont(_FONT_NAME, 7)
    canvas.setFillColor(colors.HexColor("#6B7280"))
    canvas.drawString(18 * mm, height - 12 * mm, _shape(get_bot_name(), language))
    canvas.drawRightString(width - 18 * mm, height - 12 * mm, _shape(str(doc.page), language))
    canvas.setStrokeColor(colors.HexColor("#D9E2EC"))
    canvas.setLineWidth(0.4)
    canvas.line(18 * mm, height - 15 * mm, width - 18 * mm, height - 15 * mm)
    footer = footer_text(language)
    canvas.setFillColor(colors.HexColor("#9CA3AF"))
    canvas.setFont(_FONT_NAME, 7)
    canvas.drawCentredString(width / 2, 12 * mm, _shape(footer, language))
    canvas.restoreState()


def _disclaimer_box(report: dict[str, Any], styles: dict[str, ParagraphStyle], language: str) -> Table:
    text = report.get("disclaimer") or t("not_financial_advice", language)
    table = Table([[Paragraph(_esc(text, language), styles["small"])]], colWidths=[174 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF8E6")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#F0B429")),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _idea_story(report: dict[str, Any], language: str) -> list[Any]:
    styles = _styles(language)
    currency = report.get("currency", "TND")
    financials = report.get("financials", {})
    story: list[Any] = [
        _p(get_bot_name(), styles["subtitle"], language),
        _p(t("report_title_idea", language), styles["title"], language),
        _p(
            f"{t('generated_on', language)} {datetime.now().strftime('%Y-%m-%d %H:%M')}  ·  "
            f"{preset_label(report.get('preset', 'tunisia'), language)}  ·  {currency}",
            styles["subtitle"],
            language,
        ),
        Spacer(1, 5 * mm),
    ]
    title = report.get("title") or report.get("sector_label", "")
    story.append(_p(title, styles["h2"], language))
    metrics = [
        ["Metric", "Value"],
        [t("overall_score", language), f"{report.get('overall_score', 0):.1f}/100"],
        [t("success_probability", language), f"{report.get('success_probability', 0):.1f}%"],
        [t("confidence", language), f"{report.get('confidence', 0):.0f}%"],
        [t("roi", language), f"{financials.get('roi_percent', 0):.1f}%"],
        [t("breakeven", language), f"{financials.get('breakeven_months', 0):.0f} {t('months', language)}"],
        [t("profit", language) + " (realistic)", _fmt_money(financials.get("monthly_profit"), currency) + " / " + t("month", language)],
    ]
    story.append(_table(metrics, styles, language, col_widths=[62 * mm, 112 * mm]))
    story.append(Spacer(1, 2 * mm))
    story.append(_p(f"{t('verdict', language)}: {report.get('verdict', '')}", styles["body"], language))
    story.append(_disclaimer_box(report, styles, language))

    story += _section(t("top_drivers", language), styles, language)
    story += _bullet_list(
        [f"{t('positive_drivers', language)}: " + ", ".join(report.get("positive_drivers", [])[:3])]
        + [f"{t('negative_drivers', language)}: " + ", ".join(report.get("negative_drivers", [])[:3])],
        styles,
        language,
    )

    story += _section(t("scores", language), styles, language)
    chart = _score_chart(report, language)
    if chart is not None:
        story.append(chart)
        story.append(Spacer(1, 2 * mm))
    rows = [["Criterion", "Score", "Weight"]]
    for item in report.get("criteria", []):
        rows.append([item["label"], f"{item['score']:.1f}/10", f"{item['weight']*100:.0f}%"])
    story.append(_table(rows, styles, language, col_widths=[112 * mm, 30 * mm, 32 * mm]))

    story += _section(t("financials", language), styles, language)
    assumptions = financials.get("assumptions", {})
    finance_rows = [
        ["Assumption", "Value"],
        [t("price", language), _fmt_money(assumptions.get("price"), currency)],
        [t("expected_customers", language), f"{assumptions.get('expected_customers', 0):,.0f}"],
        [t("monthly_revenue", language), _fmt_money(financials.get("monthly_revenue"), currency)],
        [t("costs", language), _fmt_money(financials.get("monthly_fixed_costs"), currency)],
        [t("marketing", language), _fmt_money(financials.get("monthly_marketing"), currency)],
        [t("profit", language), _fmt_money(financials.get("monthly_profit"), currency)],
    ]
    story.append(_table(finance_rows, styles, language, col_widths=[88 * mm, 86 * mm]))
    scenario_rows = [["Scenario", "Monthly profit", "ROI", "Break-even"]]
    for key in ("pessimistic", "realistic", "optimistic"):
        scenario = report.get("scenarios", {}).get(key, {})
        scenario_rows.append(
            [
                t(key, language),
                _fmt_money(scenario.get("monthly_profit"), currency),
                f"{scenario.get('roi_percent', 0):.1f}%",
                f"{scenario.get('breakeven_months', 0):.0f} " + t("months", language),
            ]
        )
    story += _section(t("profit_range", language), styles, language)
    story.append(_table(scenario_rows, styles, language, col_widths=[50 * mm, 48 * mm, 38 * mm, 38 * mm]))

    story += _section(t("swot", language), styles, language)
    swot = report.get("swot", {})
    swot_rows = [
        [t("strengths", language), t("weaknesses", language)],
        ["\n".join("• " + item for item in swot.get("strengths", [])), "\n".join("• " + item for item in swot.get("weaknesses", []))],
        [t("opportunities", language), t("threats", language)],
        ["\n".join("• " + item for item in swot.get("opportunities", [])), "\n".join("• " + item for item in swot.get("threats", []))],
    ]
    story.append(_table(swot_rows, styles, language, col_widths=[87 * mm, 87 * mm], header=False))

    story += _section(t("risks_actions", language), styles, language)
    story += _bullet_list(report.get("key_risks", [])[:5], styles, language)
    mod_rows = [["Action", "Priority", "Score gain", "ROI gain"]]
    for item in report.get("modifications", [])[:6]:
        mod_rows.append(
            [
                item.get("action", ""),
                t(item.get("priority", "medium"), language),
                f"+{item.get('score_gain', 0):.1f}",
                f"+{item.get('roi_gain', 0):.0f}%",
            ]
        )
    story.append(_table(mod_rows, styles, language, col_widths=[91 * mm, 26 * mm, 28 * mm, 29 * mm]))

    story += _section(t("target_segment", language), styles, language)
    story.append(_p(report.get("target_segment", ""), styles["body"], language))
    story += _section(t("go_to_market", language), styles, language)
    story += _bullet_list(report.get("go_to_market", []), styles, language)

    story += [Spacer(1, 3 * mm), _p(t("analysis_method", language), styles["small"], language)]
    story.append(_p(t("pdf_note", language), styles["small"], language))
    return story


def _business_story(report: dict[str, Any], language: str) -> list[Any]:
    styles = _styles(language)
    currency = report.get("currency", "TND")
    kpis = report.get("kpis", {})
    story: list[Any] = [
        _p(get_bot_name(), styles["subtitle"], language),
        _p(t("report_title_business", language), styles["title"], language),
        _p(
            f"{t('generated_on', language)} {datetime.now().strftime('%Y-%m-%d %H:%M')}  ·  "
            f"{preset_label(report.get('preset', 'tunisia'), language)}  ·  {currency}",
            styles["subtitle"],
            language,
        ),
        Spacer(1, 5 * mm),
        _p(report.get("title", ""), styles["h2"], language),
        _p(report.get("summary", ""), styles["body"], language),
        _disclaimer_box(report, styles, language),
    ]
    rows = [
        ["KPI", "Value"],
        [t("revenue", language) + " (last month)", _fmt_money(kpis.get("last_month_revenue"), currency)],
        [t("profit", language) + " (last month)", _fmt_money(kpis.get("last_month_profit"), currency)],
        [t("margin", language), f"{kpis.get('net_margin_pct', 0):.1f}%"],
        [t("growth", language), f"{kpis.get('monthly_growth_pct', 0):.1f}%"],
        [t("churn", language), f"{kpis.get('churn_pct', 0):.1f}%"],
        [t("cac", language), _fmt_money(kpis.get("cac"), currency)],
        [t("ltv", language), _fmt_money(kpis.get("ltv"), currency)],
        [t("ltv_cac", language), f"{kpis.get('ltv_cac', 0):.1f}×" if kpis.get("ltv_cac") else "—"],
        [t("marketing_ratio", language), f"{kpis.get('marketing_ratio_pct', 0):.1f}%"],
        [t("confidence", language), f"{report.get('confidence', 0):.0f}%"],
    ]
    story += _section(t("kpis", language), styles, language)
    story.append(_table(rows, styles, language, col_widths=[88 * mm, 86 * mm]))

    story += _section(t("benchmarks", language), styles, language)
    bench_rows = [["Metric", "Actual", "Benchmark", "Status"]]
    for item in report.get("benchmarks", []):
        unit = item.get("unit")
        actual = item.get("actual")
        benchmark = item.get("benchmark")
        if actual is None:
            actual_text = "—"
        elif unit == "%":
            actual_text = f"{actual:.1f}%"
        elif unit == "currency":
            actual_text = _fmt_money(actual, currency)
        else:
            actual_text = f"{actual:.1f}×"
        if benchmark is None:
            benchmark_text = "—"
        elif unit == "%":
            benchmark_text = f"{benchmark:.1f}%"
        elif unit == "currency":
            benchmark_text = _fmt_money(benchmark, currency)
        else:
            benchmark_text = f"{benchmark:.1f}×"
        bench_rows.append([item["label"], actual_text, benchmark_text, t(item.get("status", "watch"), language)])
    story.append(_table(bench_rows, styles, language, col_widths=[72 * mm, 35 * mm, 35 * mm, 32 * mm]))

    story += _section(t("detected_problems", language), styles, language)
    problems = report.get("problems", [])
    if problems:
        for problem in problems:
            story.append(_p(f"• [{problem.get('severity', 'medium').upper()}] {problem.get('title')}: {problem.get('detail')}", styles["body"], language))
    else:
        story.append(_p(t("no_problems", language), styles["body"], language))

    story += _section(t("prioritized_actions", language), styles, language)
    action_rows = [["Action", "Priority", "Impact / month"]]
    for action in report.get("actions", [])[:6]:
        action_rows.append(
            [
                action.get("action", ""),
                t(action.get("priority", "medium"), language),
                _fmt_money(action.get("impact_monthly"), currency),
            ]
        )
    story.append(_table(action_rows, styles, language, col_widths=[108 * mm, 28 * mm, 38 * mm]))

    story += _section(t("forecast", language), styles, language)
    chart = _forecast_chart(report, language)
    if chart is not None:
        story.append(chart)
        story.append(_p(f"{t('forecast_method', language)}: {report.get('forecast', {}).get('method', '')}", styles["small"], language))
    else:
        story.append(_p((report.get("forecast") or {}).get("error", "Forecast unavailable."), styles["small"], language))

    notes = (report.get("data_quality") or {}).get("notes") or []
    story += _section(t("data_quality", language), styles, language)
    story += _bullet_list(notes or ["—"], styles, language)
    story.append(_p(t("pdf_note", language), styles["small"], language))
    return story


def build_report_pdf(report: dict[str, Any]) -> bytes:
    """Build a localized PDF and return it as bytes (safe for st.download_button)."""
    language = normalize_language(report.get("language", "en"))
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=21 * mm,
        bottomMargin=20 * mm,
        title=f"{get_bot_name()} — {t('report_title_idea' if report.get('mode') == 'idea' else 'report_title_business', language)}",
        author=get_bot_name(),
    )
    story = _idea_story(report, language) if report.get("mode") == "idea" else _business_story(report, language)
    doc.build(
        story,
        onFirstPage=lambda canvas, doc: _on_page(canvas, doc, language),
        onLaterPages=lambda canvas, doc: _on_page(canvas, doc, language),
    )
    return buffer.getvalue()
