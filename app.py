"""CEO AI — Streamlit conversational interface.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.agent import AgentState, process_message, start_conversation, update_preferences
from src.business_analysis import analyze_business, build_manual_snapshot
from src.config import (
    currency_for,
    footer_text,
    get_bot_name,
    get_config,
    get_language_name,
    get_paths,
    get_preset,
    preset_label,
    preset_names,
    sector_label,
    sector_names,
    supported_languages,
)
from src.i18n import t
from src.nlg import llm_status, report_summary_markdown
from src.report import build_report_pdf
from src.scoring import load_models, simulate_what_if
from src.storage import get_evaluation, list_evaluations
from src.utils import format_money, normalize_language, safe_float

BOT_NAME = get_bot_name()
CONFIG = get_config()
DEFAULT_LANGUAGE = normalize_language(CONFIG.get("default_language", "en"))

st.set_page_config(
    page_title=BOT_NAME,
    page_icon=str(CONFIG.get("ui", {}).get("page_icon", "🧠")),
    layout=str(CONFIG.get("ui", {}).get("layout", "wide")),
    initial_sidebar_state="expanded",
)


def _inject_css(language: str) -> None:
    rtl_css = ""
    if language == "ar":
        rtl_css = """
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] li,
        [data-testid="stChatMessage"] h1,
        [data-testid="stChatMessage"] h2,
        [data-testid="stChatMessage"] h3 { direction: rtl; text-align: right; }
        """
    st.markdown(
        f"""
        <style>
        .block-container {{ padding-bottom: 4.2rem; }}
        [data-testid="stChatMessage"] {{ border-radius: 0.85rem; }}
        [data-testid="stMetric"] {{
            background: rgba(11, 95, 255, 0.045);
            border: 1px solid rgba(11, 95, 255, 0.12);
            border-radius: 0.75rem;
            padding: 0.65rem 0.8rem;
        }}
        .app-footer {{
            position: fixed;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: 999999;
            text-align: center;
            font-size: 0.72rem;
            line-height: 1.2rem;
            color: #9ca3af;
            padding: 4px 8px 6px 8px;
            background: rgba(255, 255, 255, 0.90);
            pointer-events: none;
        }}
        [data-testid="stBottom"] {{ padding-bottom: 1.75rem !important; }}
        [data-testid="stBottomBlockContainer"] {{ padding-bottom: 1.75rem !important; }}
        @media (prefers-color-scheme: dark) {{
            .app-footer {{ background: rgba(14, 17, 23, 0.90); color: #9ca3af; }}
        }}
        {rtl_css}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_footer(language: str) -> None:
    text = footer_text(language)
    st.markdown(
        f'<div class="app-footer">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def _reset_workspace() -> None:
    for key in (
        "agent",
        "messages",
        "current_mode",
        "idea_result",
        "business_result",
        "business_preview",
        "loaded_report",
        "last_upload_signature",
    ):
        st.session_state.pop(key, None)


def _ensure_agent(mode: str, language: str, preset_name: str) -> None:
    existing = st.session_state.get("agent")
    if existing is None or st.session_state.get("current_mode") != mode:
        agent, welcome = start_conversation(language, preset_name, mode)
        st.session_state.agent = agent
        st.session_state.messages = [{"role": "assistant", "content": welcome}]
        st.session_state.current_mode = mode
        st.session_state.idea_result = None
        st.session_state.business_result = None
    else:
        update_preferences(existing, language=language, preset_name=preset_name)


def _chat_message(role: str, content: str) -> None:
    if role == "assistant":
        with st.chat_message("assistant", avatar=str(CONFIG.get("ui", {}).get("page_icon", "🧠"))):
            st.markdown(f"**{BOT_NAME}**")
            st.markdown(content)
    else:
        with st.chat_message("user", avatar="👤"):
            st.markdown(content)


def _radar_figure(report: dict[str, Any], title: str | None = None) -> go.Figure:
    criteria = report.get("criteria", [])
    labels = [item["label"] for item in criteria]
    values = [item["score"] for item in criteria]
    if labels:
        labels.append(labels[0])
        values.append(values[0])
    fig = go.Figure(
        go.Scatterpolar(
            r=values,
            theta=labels,
            fill="toself",
            name=title or t("overall_score", report.get("language", "en")),
            line=dict(color="#0B5FFF", width=2),
            fillcolor="rgba(11,95,255,0.16)",
        )
    )
    fig.update_layout(
        template="plotly_white",
        polar=dict(radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(size=9))),
        showlegend=False,
        height=430,
        margin=dict(l=35, r=35, t=35, b=35),
    )
    return fig


def _scenario_figure(report: dict[str, Any]) -> go.Figure:
    lang = report.get("language", "en")
    currency = report.get("currency", "TND")
    scenarios = report.get("scenarios", {})
    labels = [t(key, lang) for key in ("pessimistic", "realistic", "optimistic")]
    values = [scenarios.get(key, {}).get("monthly_profit", 0) for key in ("pessimistic", "realistic", "optimistic")]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=["#D64545", "#0B5FFF", "#2F855A"],
            text=[format_money(value, currency) for value in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        template="plotly_white",
        yaxis_title=currency,
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
    )
    return fig


def _explanations_figure(report: dict[str, Any]) -> go.Figure | None:
    explanations = report.get("explanations", [])[:8]
    if not explanations:
        return None
    frame = pd.DataFrame(explanations).sort_values("contribution")
    colors = ["#D64545" if value < 0 else "#2F855A" for value in frame["contribution"]]
    fig = go.Figure(go.Bar(x=frame["contribution"], y=frame["label"], orientation="h", marker_color=colors))
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Contribution to success probability",
        height=max(260, 34 * len(frame)),
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
    )
    return fig


def _forecast_figure(report: dict[str, Any]) -> go.Figure | None:
    forecast = report.get("forecast") or {}
    records = forecast.get("revenue") or []
    if not records:
        return None
    frame = pd.DataFrame(records)
    frame["date"] = pd.to_datetime(frame["date"])
    fig = go.Figure()
    historical = pd.DataFrame(report.get("data_preview") or [])
    if not historical.empty:
        historical["date"] = pd.to_datetime(historical["date"])
        fig.add_trace(
            go.Scatter(
                x=historical["date"],
                y=historical["revenue"],
                mode="lines+markers",
                name=t("revenue", report.get("language", "en")),
                line=dict(color="#52606D", width=2),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["upper"],
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["lower"],
            fill="tonexty",
            fillcolor="rgba(11,95,255,0.13)",
            line=dict(width=0),
            name=t("confidence_band", report.get("language", "en")),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["forecast"],
            mode="lines+markers",
            name=t("forecast", report.get("language", "en")),
            line=dict(color="#0B5FFF", width=2.4),
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=360,
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode="x unified",
    )
    return fig


def _pdf_download(report: dict[str, Any], *, key: str) -> None:
    lang = report.get("language", "en")
    try:
        payload = build_report_pdf(report)
        st.download_button(
            t("download_pdf", lang),
            data=payload,
            file_name=f"{BOT_NAME.replace(' ', '_').lower()}_report_{report.get('report_id') or 'draft'}.pdf",
            mime="application/pdf",
            key=key,
        )
    except Exception as exc:
        st.error(f"{t('error', lang)}: {exc}")


def _render_overview(report: dict[str, Any]) -> None:
    lang = report.get("language", "en")
    financials = report.get("financials", {})
    currency = report.get("currency", "TND")
    verdict_key = report.get("verdict_key")
    if verdict_key == "succeed":
        st.success(report.get("verdict", ""))
    elif verdict_key == "risky":
        st.warning(report.get("verdict", ""))
    else:
        st.error(report.get("verdict", ""))
    cols = st.columns(5)
    cols[0].metric(t("overall_score", lang), f"{report.get('overall_score', 0):.1f}/100")
    cols[1].metric(t("success_probability", lang), f"{report.get('success_probability', 0):.1f}%")
    cols[2].metric(t("confidence", lang), f"{report.get('confidence', 0):.0f}%")
    cols[3].metric(t("roi", lang), f"{financials.get('roi_percent', 0):.1f}%")
    cols[4].metric(t("breakeven", lang), f"{financials.get('breakeven_months', 0):.0f} " + t("months", lang))
    st.progress(min(max(report.get("overall_score", 0) / 100, 0.0), 1.0))
    st.caption(
        f"{report.get('sector_label', '')} · {report.get('location', '')} · "
        f"{preset_label(report.get('preset', 'tunisia'), lang)} · {currency}"
    )
    if not report.get("model_available", False):
        st.warning(t("model_missing", lang))

    left, right = st.columns([1.15, 1.0])
    with left:
        st.plotly_chart(_radar_figure(report), use_container_width=True, key=f"radar_{report.get('report_id')}")
    with right:
        st.markdown(f"#### {t('top_drivers', lang)}")
        st.markdown(f"**{t('positive_drivers', lang)}**")
        for item in report.get("positive_drivers", [])[:3]:
            st.markdown(f"- {item}")
        st.markdown(f"**{t('negative_drivers', lang)}**")
        for item in report.get("negative_drivers", [])[:3]:
            st.markdown(f"- {item}")
        st.caption(t("explanation_note", lang))
        segment = report.get("segment")
        if segment:
            st.info(f"ML segment: **{segment.get('name', '')}**")
    st.warning(report.get("disclaimer", ""))

    chart = _explanations_figure(report)
    if chart is not None:
        st.plotly_chart(chart, use_container_width=True, key=f"explain_{report.get('report_id')}")


def _render_scores(report: dict[str, Any]) -> None:
    lang = report.get("language", "en")
    criteria = pd.DataFrame(report.get("criteria", []))
    if not criteria.empty:
        criteria["weight"] = (criteria["weight"] * 100).round(0).astype(int).astype(str) + "%"
        criteria["score"] = criteria["score"].map(lambda value: f"{value:.1f}/10")
        st.dataframe(
            criteria.rename(columns={"label": t("criteria", lang), "score": "Score", "weight": t("weight", lang)}),
            hide_index=True,
            use_container_width=True,
        )
    st.plotly_chart(_radar_figure(report), use_container_width=True, key=f"radar_scores_{report.get('report_id')}")


def _render_financials(report: dict[str, Any]) -> None:
    lang = report.get("language", "en")
    currency = report.get("currency", "TND")
    financials = report.get("financials", {})
    cols = st.columns(4)
    cols[0].metric(t("monthly_revenue", lang), format_money(financials.get("monthly_revenue"), currency))
    cols[1].metric(t("profit", lang), format_money(financials.get("monthly_profit"), currency))
    cols[2].metric(t("roi", lang), f"{financials.get('roi_percent', 0):.1f}%")
    cols[3].metric(t("breakeven", lang), f"{financials.get('breakeven_months', 0):.0f} " + t("months", lang))
    st.plotly_chart(_scenario_figure(report), use_container_width=True, key=f"scenario_{report.get('report_id')}")
    assumptions = financials.get("assumptions", {})
    st.caption(
        f"{t('price', lang)}: {format_money(assumptions.get('price'), currency)} · "
        f"{t('expected_customers', lang)}: {assumptions.get('expected_customers', 0):,.0f} · "
        f"{t('margin', lang)}: {assumptions.get('gross_margin_pct', 0):.1f}% · "
        f"{t('marketing', lang)}: {format_money(assumptions.get('marketing_spend'), currency)}"
    )
    st.warning(t("not_financial_advice", lang))


def _render_swot(report: dict[str, Any]) -> None:
    lang = report.get("language", "en")
    swot = report.get("swot", {})
    cols = st.columns(4)
    for column, key in zip(cols, ("strengths", "weaknesses", "opportunities", "threats")):
        with column:
            st.markdown(f"**{t(key, lang)}**")
            for item in swot.get(key, []):
                st.markdown(f"- {item}")
    st.markdown(f"#### {t('target_segment', lang)}")
    st.info(report.get("target_segment", ""))
    st.markdown(f"#### {t('go_to_market', lang)}")
    for item in report.get("go_to_market", []):
        st.markdown(f"- {item}")


def _render_actions(report: dict[str, Any]) -> None:
    lang = report.get("language", "en")
    st.markdown(f"#### {t('key_risks', lang)}")
    for item in report.get("key_risks", []):
        st.warning(item)
    st.markdown(f"#### {t('recommended_modifications', lang)}")
    modifications = report.get("modifications", [])
    if modifications:
        frame = pd.DataFrame(modifications)
        frame["priority"] = frame["priority"].map(lambda value: t(value, lang))
        st.dataframe(
            frame[
                ["action", "priority", "score_gain", "roi_gain", "cost_note"]
            ].rename(
                columns={
                    "action": t("recommended_modifications", lang),
                    "priority": t("priority", lang),
                    "score_gain": t("score_gain", lang),
                    "roi_gain": t("roi_gain", lang),
                    "cost_note": t("impact", lang),
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
    st.caption(t("answer_default", lang))


def _render_what_if(report: dict[str, Any]) -> None:
    lang = report.get("language", "en")
    currency = report.get("currency", "TND")
    inputs = report.get("created_inputs", {})
    rid = report.get("report_id") or "draft"
    st.caption(t("what_if_help", lang))
    base_price = max(1.0, safe_float(inputs.get("price"), 50))
    base_budget = max(500.0, safe_float(inputs.get("budget"), 30000))
    base_marketing = max(0.0, safe_float(inputs.get("marketing_spend"), 500))
    base_customers = max(1.0, safe_float(inputs.get("expected_customers"), 100))
    cols = st.columns(4)
    price = cols[0].slider(
        f"{t('price', lang)} ({currency})", min_value=1.0, max_value=float(max(base_price * 2.5, 100)), value=float(base_price), step=1.0, key=f"wi_price_{rid}"
    )
    marketing = cols[1].slider(
        f"{t('marketing', lang)} ({currency})", min_value=0.0, max_value=float(max(base_marketing * 2.5, 1000)), value=float(base_marketing), step=10.0, key=f"wi_marketing_{rid}"
    )
    customers = cols[2].slider(
        t("expected_customers", lang), min_value=1.0, max_value=float(max(base_customers * 2.5, 100)), value=float(base_customers), step=1.0, key=f"wi_customers_{rid}"
    )
    differentiation = cols[3].slider(
        t("differentiation", lang), min_value=1.0, max_value=10.0, value=float(safe_float(inputs.get("differentiation"), 6)), step=0.5, key=f"wi_diff_{rid}"
    )
    budget = st.slider(
        f"{t('budget', lang)} ({currency})", min_value=500.0, max_value=float(max(base_budget * 3, 10000)), value=float(base_budget), step=100.0, key=f"wi_budget_{rid}"
    )
    changes = {
        "price": price,
        "marketing_spend": marketing,
        "expected_customers": customers,
        "differentiation": differentiation,
        "budget": budget,
    }
    try:
        simulation = simulate_what_if(inputs, changes, report.get("preset", "tunisia"), lang)
        scenario = simulation["scenario"]
        cols = st.columns(4)
        cols[0].metric(
            t("overall_score", lang),
            f"{scenario['overall_score']:.1f}/100",
            delta=f"{simulation['delta_score']:+.1f}",
        )
        cols[1].metric(
            t("success_probability", lang),
            f"{scenario['success_probability']:.1f}%",
            delta=f"{simulation['delta_probability']:+.1f}",
        )
        cols[2].metric(
            t("roi", lang),
            f"{scenario['financials']['roi_percent']:.1f}%",
            delta=f"{simulation['delta_roi']:+.1f}",
        )
        cols[3].metric(
            t("profit", lang),
            format_money(scenario["financials"]["monthly_profit"], currency),
            delta=format_money(simulation["delta_profit"], currency),
        )
    except Exception as exc:
        st.error(f"{t('error', lang)}: {exc}")


def render_idea_report(report: dict[str, Any]) -> None:
    lang = report.get("language", "en")
    st.divider()
    st.subheader(f"📊 {report.get('title', '')}")
    tabs = st.tabs(
        [
            t("overview", lang),
            t("scores", lang),
            t("financials", lang),
            t("swot", lang),
            t("risks_actions", lang),
            t("what_if", lang),
        ]
    )
    with tabs[0]:
        _render_overview(report)
    with tabs[1]:
        _render_scores(report)
    with tabs[2]:
        _render_financials(report)
    with tabs[3]:
        _render_swot(report)
    with tabs[4]:
        _render_actions(report)
    with tabs[5]:
        _render_what_if(report)
    _pdf_download(report, key=f"pdf_idea_{report.get('report_id') or 'draft'}")


def render_business_report(report: dict[str, Any]) -> None:
    lang = report.get("language", "en")
    currency = report.get("currency", "TND")
    kpis = report.get("kpis", {})
    st.divider()
    st.subheader(f"📈 {report.get('title', '')}")
    cols = st.columns(5)
    cols[0].metric(t("revenue", lang), format_money(kpis.get("last_month_revenue"), currency))
    cols[1].metric(t("profit", lang), format_money(kpis.get("last_month_profit"), currency))
    cols[2].metric(t("margin", lang), f"{kpis.get('net_margin_pct', 0):.1f}%")
    cols[3].metric(t("growth", lang), f"{kpis.get('monthly_growth_pct', 0):.1f}%")
    cols[4].metric(t("confidence", lang), f"{report.get('confidence', 0):.0f}%")
    st.info(report.get("summary", ""))
    tabs = st.tabs(
        [
            t("kpis", lang),
            t("benchmarks", lang),
            t("risks_actions", lang),
            t("forecast", lang),
            t("data_quality", lang),
        ]
    )
    with tabs[0]:
        detail = [
            (t("churn", lang), f"{kpis.get('churn_pct', 0):.1f}%"),
            (t("cac", lang), format_money(kpis.get("cac"), currency)),
            (t("ltv", lang), format_money(kpis.get("ltv"), currency)),
            (t("ltv_cac", lang), f"{kpis.get('ltv_cac', 0):.1f}×" if kpis.get("ltv_cac") else "—"),
            (t("marketing_ratio", lang), f"{kpis.get('marketing_ratio_pct', 0):.1f}%"),
            (t("avg_order_value", lang), format_money(kpis.get("avg_order_value"), currency)),
        ]
        st.table(pd.DataFrame(detail, columns=[t("kpis", lang), t("actual", lang)]))
    with tabs[1]:
        frame = pd.DataFrame(report.get("benchmarks", []))
        if not frame.empty:
            frame["status"] = frame["status"].map(lambda value: t(value, lang) if value in {"good", "watch", "bad"} else "—")
            frame["actual"] = frame["actual"].map(lambda value: "—" if pd.isna(value) else f"{float(value):.2f}")
            frame["benchmark"] = frame["benchmark"].map(lambda value: "—" if pd.isna(value) else f"{float(value):.2f}")
            frame = frame[["label", "actual", "benchmark", "status"]].rename(
                columns={
                    "label": t("kpis", lang),
                    "actual": t("actual", lang),
                    "benchmark": t("benchmark", lang),
                    "status": t("status", lang),
                }
            )
            st.dataframe(frame, hide_index=True, use_container_width=True)
    with tabs[2]:
        st.markdown(f"#### {t('detected_problems', lang)}")
        problems = report.get("problems", [])
        if problems:
            for problem in problems:
                icon = "🔴" if problem.get("severity") == "high" else "🟠"
                st.markdown(f"{icon} **{problem.get('title')}** — {problem.get('detail')}")
        else:
            st.success(t("no_problems", lang))
        st.markdown(f"#### {t('prioritized_actions', lang)}")
        actions = pd.DataFrame(report.get("actions", []))
        if not actions.empty:
            actions["priority"] = actions["priority"].map(lambda value: t(value, lang))
            actions["impact_monthly"] = actions["impact_monthly"].map(lambda value: format_money(value, currency))
            actions = actions[["action", "priority", "impact_monthly", "rationale"]].rename(
                columns={
                    "action": t("prioritized_actions", lang),
                    "rationale": t("rationale", lang),
                    "priority": t("priority", lang),
                    "impact_monthly": t("estimated_impact", lang),
                }
            )
            st.dataframe(actions, hide_index=True, use_container_width=True)
    with tabs[3]:
        chart = _forecast_figure(report)
        if chart is not None:
            st.plotly_chart(chart, use_container_width=True, key=f"forecast_{report.get('report_id')}")
            st.caption(f"{t('forecast_method', lang)}: {(report.get('forecast') or {}).get('method', '')}")
        else:
            st.warning((report.get("forecast") or {}).get("error", "Forecast unavailable."))
    with tabs[4]:
        quality = report.get("data_quality", {})
        st.write(
            f"{t('months', lang)}: **{quality.get('months', '—')}** · "
            f"{quality.get('period_start', '')} → {quality.get('period_end', '')}"
        )
        for note in quality.get("notes", []):
            st.warning(note)
        st.dataframe(pd.DataFrame(report.get("data_preview", [])), hide_index=True, use_container_width=True)
    st.warning(report.get("disclaimer", ""))
    _pdf_download(report, key=f"pdf_business_{report.get('report_id') or 'draft'}")


def _read_uploaded_file(uploaded_file: Any) -> pd.DataFrame:
    name = str(getattr(uploaded_file, "name", "upload.csv")).lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)


def _run_business_analysis(
    frame: pd.DataFrame,
    *,
    sector: str,
    preset_name: str,
    language: str,
    manual: bool = False,
) -> None:
    report = analyze_business(
        frame,
        sector=sector,
        preset_name=preset_name,
        language=language,
        periods=6,
        manual_snapshot=manual,
    )
    from src.storage import save_evaluation

    try:
        report["report_id"] = save_evaluation(report)
    except Exception:
        report["report_id"] = None
    st.session_state.business_result = report
    st.session_state.messages.append(
        {"role": "assistant", "content": report_summary_markdown(report, language)}
    )


def render_idea_mode(language: str, preset_name: str) -> None:
    agent: AgentState = st.session_state.agent
    for message in st.session_state.messages:
        _chat_message(message["role"], message["content"])
    if st.session_state.get("idea_result"):
        render_idea_report(st.session_state.idea_result)
    prompt = st.chat_input(t("chat_placeholder", language))
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        turn = process_message(agent, prompt)
        st.session_state.messages.append({"role": "assistant", "content": turn.response})
        if turn.ready and turn.result:
            st.session_state.idea_result = turn.result
        st.rerun()


def render_business_mode(language: str, preset_name: str, uploaded_file: Any, sector: str) -> None:
    st.caption(t("upload_help", language))
    if uploaded_file is None:
        st.info(t("business_greeting", language))
    else:
        try:
            preview = _read_uploaded_file(uploaded_file)
            st.markdown(f"#### {t('data_preview', language)}")
            st.dataframe(preview.head(8), use_container_width=True, hide_index=True)
            if st.button(t("analyze", language), type="primary", key="analyze_upload"):
                _run_business_analysis(
                    preview,
                    sector=sector,
                    preset_name=preset_name,
                    language=language,
                    manual=False,
                )
                st.rerun()
        except Exception as exc:
            st.error(f"{t('error', language)}: {exc}")

    with st.expander(t("manual_snapshot", language), expanded=uploaded_file is None):
        preset = get_preset(preset_name)
        currency = currency_for(preset_name)
        st.caption(t("monthly_snapshot_hint", language))
        with st.form("manual_business_form"):
            col1, col2 = st.columns(2)
            revenue = col1.number_input(
                f"{t('monthly_revenue', language)} ({currency})",
                min_value=0.0,
                value=float(preset.get("typical_price", 50) * preset.get("typical_customers", 150)),
                step=100.0,
            )
            cost = col2.number_input(
                f"{t('monthly_cost', language)} ({currency})",
                min_value=0.0,
                value=float(preset.get("typical_price", 50) * preset.get("typical_customers", 150) * 0.72),
                step=100.0,
            )
            col3, col4 = st.columns(2)
            customers = col3.number_input(t("customers", language), min_value=1.0, value=float(preset.get("typical_customers", 150)), step=1.0)
            marketing = col4.number_input(
                f"{t('monthly_marketing', language)} ({currency})",
                min_value=0.0,
                value=float(preset.get("typical_marketing", 500)),
                step=50.0,
            )
            growth = st.slider(t("monthly_growth_pct", language), min_value=-20.0, max_value=30.0, value=3.0, step=0.5)
            submitted = st.form_submit_button(t("analyze_manual", language), type="primary")
        if submitted:
            frame, _ = build_manual_snapshot(revenue, cost, customers, marketing, growth)
            _run_business_analysis(
                frame,
                sector=sector,
                preset_name=preset_name,
                language=language,
                manual=True,
            )
            st.rerun()

    for message in st.session_state.messages:
        _chat_message(message["role"], message["content"])
    if st.session_state.get("business_result"):
        render_business_report(st.session_state.business_result)
    prompt = st.chat_input(t("chat_placeholder", language))
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        if st.session_state.get("business_result"):
            from src.nlg import answer_followup

            response = answer_followup(st.session_state.business_result, prompt, language)
        else:
            response = t("business_greeting", language)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


def render_history_mode(language: str) -> None:
    st.subheader(f"🗂 {t('history_title', language)}")
    records = list_evaluations(limit=100)
    if not records:
        st.info(t("no_history", language))
        return
    for record in records:
        record["label"] = (
            f"#{record['id']} · {str(record['title'])[:65]} · {str(record['created_at'])[:16]}"
        )
    frame = pd.DataFrame(records)
    st.dataframe(
        frame[["id", "created_at", "mode", "title", "overall_score", "roi_percent"]].rename(
            columns={
                "id": "ID",
                "created_at": t("saved_at", language),
                "mode": t("mode", language),
                "title": t("idea", language),
                "overall_score": t("overall_score", language),
                "roi_percent": t("roi", language),
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    selected_id = st.selectbox(
        t("select_evaluation", language),
        options=[int(row["id"]) for row in records],
        format_func=lambda value: next(row["label"] for row in records if int(row["id"]) == int(value)),
    )
    if selected_id:
        loaded = get_evaluation(int(selected_id))
        if loaded:
            if loaded.get("mode") == "idea":
                render_idea_report(loaded)
            else:
                render_business_report(loaded)

    st.divider()
    st.subheader(f"⚖️ {t('compare_title', language)}")
    st.caption(t("select_2_3", language))
    compare_ids = st.multiselect(
        t("select_2_3", language),
        options=[int(row["id"]) for row in records],
        max_selections=3,
        format_func=lambda value: next(row["label"] for row in records if int(row["id"]) == int(value)),
    )
    if len(compare_ids) >= 2:
        reports = [get_evaluation(report_id) for report_id in compare_ids]
        reports = [report for report in reports if report]
        if len(reports) >= 2:
            rows = []
            for report in reports:
                financials = report.get("financials") or {}
                kpis = report.get("kpis") or {}
                rows.append(
                    {
                        t("idea", language): (report.get("title") or "")[:55],
                        t("overall_score", language): report.get("overall_score"),
                        t("success_probability", language): report.get("success_probability"),
                        t("roi", language): financials.get("roi_percent", kpis.get("net_margin_pct")),
                        t("confidence", language): report.get("confidence"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            idea_reports = [report for report in reports if report.get("mode") == "idea"]
            if len(idea_reports) >= 2:
                fig = go.Figure()
                for report in idea_reports:
                    criteria = report.get("criteria", [])
                    labels = [item["label"] for item in criteria]
                    values = [item["score"] for item in criteria]
                    if labels:
                        labels.append(labels[0])
                        values.append(values[0])
                    fig.add_trace(
                        go.Scatterpolar(
                            r=values,
                            theta=labels,
                            fill="toself",
                            name=str(report.get("title", ""))[:40],
                            opacity=0.62,
                        )
                    )
                fig.update_layout(
                    template="plotly_white",
                    polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                    height=520,
                )
                st.plotly_chart(fig, use_container_width=True, key="compare_radar")
            else:
                st.info("Radar comparison is available for idea evaluations.")


# ---------------------------------------------------------------------------
# Sidebar and state bootstrap
# ---------------------------------------------------------------------------
st.sidebar.title(f"🧠 {BOT_NAME}")
st.sidebar.caption(t("tagline", DEFAULT_LANGUAGE))

mode_labels = {"idea": "mode_idea", "business": "mode_business", "history": "mode_history"}
if "ui_language" not in st.session_state:
    st.session_state.ui_language = DEFAULT_LANGUAGE
if "ui_preset" not in st.session_state:
    st.session_state.ui_preset = preset_names()[0]
if "ui_mode" not in st.session_state:
    st.session_state.ui_mode = "idea"

language = st.sidebar.selectbox(
    t("language", st.session_state.ui_language),
    options=list(supported_languages()),
    format_func=get_language_name,
    key="ui_language",
)
language = normalize_language(language)
preset_name = st.sidebar.selectbox(
    t("market_preset", language),
    options=preset_names(),
    format_func=lambda value: preset_label(value, language),
    key="ui_preset",
)
mode = st.sidebar.radio(
    t("mode", language),
    options=list(mode_labels.keys()),
    format_func=lambda value: t(mode_labels[value], language),
    key="ui_mode",
)
st.sidebar.divider()
if st.sidebar.button(f"🔄 {t('reset_chat', language)}", use_container_width=True):
    _reset_workspace()
    st.rerun()

status = llm_status()
status_labels = {"offline": t("llm_offline", language), "online": t("llm_online", language), "error": t("llm_error", language)}
st.sidebar.caption(f"LLM: {status_labels.get(status, status)}")
models = load_models()
st.sidebar.caption(f"Models: {'ready' if models.get('available') else 'missing'}")

uploaded_file = None
business_sector = "other"
if mode == "business":
    st.sidebar.divider()
    uploaded_file = st.sidebar.file_uploader(
        t("upload_data", language),
        type=["csv", "xlsx", "xls"],
        help=t("upload_help", language),
    )
    business_sector = st.sidebar.selectbox(
        t("sector", language),
        options=sector_names(),
        format_func=lambda value: sector_label(value, language),
        index=sector_names().index("other"),
    )

_inject_css(language)

# Start/reset the correct conversation when mode changes.
if mode != "history":
    _ensure_agent(mode, language, preset_name)
    if "messages" not in st.session_state:
        st.session_state.messages = []

if mode == "idea":
    render_idea_mode(language, preset_name)
elif mode == "business":
    render_business_mode(language, preset_name, uploaded_file, business_sector)
else:
    render_history_mode(language)

_render_footer(language)
