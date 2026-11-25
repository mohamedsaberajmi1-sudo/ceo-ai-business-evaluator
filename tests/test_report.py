from src.report import build_report_pdf
from src.scoring import evaluate_idea

SAMPLE = {
    "description": "A local e-commerce store for sustainable home products.",
    "sector": "ecommerce",
    "target_market": "b2c",
    "target_customers": "urban households",
    "location": "Sfax",
    "budget": 25000,
    "team_size": 3,
    "price": 60,
    "expected_customers": 300,
    "revenue_model": "hybrid",
    "competitor_intensity": 8,
    "differentiation": 6,
    "timeline_months": 6,
}


def test_pdf_export_returns_valid_pdf():
    report = evaluate_idea(SAMPLE, "tunisia", "en", include_actions=True)
    payload = build_report_pdf(report)
    assert payload.startswith(b"%PDF")
    assert len(payload) > 3000


def test_pdf_export_supports_arabic():
    report = evaluate_idea(SAMPLE, "tunisia", "ar", include_actions=False)
    payload = build_report_pdf(report)
    assert payload.startswith(b"%PDF")
    assert len(payload) > 2000
