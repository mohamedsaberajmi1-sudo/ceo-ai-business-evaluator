from src.scoring import evaluate_idea, simulate_what_if

SAMPLE = {
    "description": "A B2B subscription platform that helps small restaurants reduce food waste.",
    "sector": "saas",
    "target_market": "b2b",
    "target_customers": "independent restaurants in Tunis",
    "location": "Tunis",
    "budget": 45000,
    "team_size": 4,
    "price": 130,
    "expected_customers": 90,
    "revenue_model": "subscription",
    "competitor_intensity": 7,
    "differentiation": 8,
    "timeline_months": 8,
}


def test_evaluate_idea_contract_and_ranges():
    report = evaluate_idea(SAMPLE, "tunisia", "en")
    assert 0 <= report["overall_score"] <= 100
    assert 0 <= report["success_probability"] <= 100
    assert 0 <= report["confidence"] <= 100
    assert report["verdict_key"] in {"succeed", "risky", "unlikely"}
    assert report["verdict"]
    assert len(report["subscores"]) == 12
    assert len(report["criteria"]) == 12
    for key, score in report["subscores"].items():
        assert 1 <= score <= 10, key
    assert report["financials"]["roi_percent"] is not None
    assert 1 <= report["financials"]["breakeven_months"] <= 60
    assert set(report["scenarios"]) == {"pessimistic", "realistic", "optimistic"}
    assert report["key_risks"]
    assert report["modifications"]
    assert isinstance(report["swot"]["strengths"], list)
    assert report["target_segment"]
    assert report["go_to_market"]
    assert "not financial advice" in report["disclaimer"].lower()
    assert report["created_inputs"]["sector"] == "saas"


def test_evaluate_idea_is_localized():
    report_fr = evaluate_idea(SAMPLE, "tunisia", "fr")
    report_ar = evaluate_idea(SAMPLE, "tunisia", "ar")
    assert report_fr["language"] == "fr"
    assert report_ar["language"] == "ar"
    assert report_fr["verdict"] != report_ar["verdict"]
    assert len(report_ar["positive_drivers"]) > 0


def test_what_if_improves_score():
    simulation = simulate_what_if(
        SAMPLE,
        {
            "differentiation": 9.5,
            "competitor_intensity": 3.0,
            "budget": 65000,
        },
        "tunisia",
        "en",
    )
    assert simulation["scenario"]["overall_score"] >= simulation["base"]["overall_score"]
    assert simulation["delta_score"] >= 0
    assert simulation["scenario"]["financials"]["monthly_profit"] is not None


def test_two_presets_are_supported():
    tunisia = evaluate_idea(SAMPLE, "tunisia", "en", include_actions=False, include_explanations=False)
    generic = evaluate_idea({**SAMPLE, "location": "Paris"}, "generic", "en", include_actions=False, include_explanations=False)
    assert tunisia["currency"] == "TND"
    assert generic["currency"] == "USD"
    assert tunisia["overall_score"] != generic["overall_score"]
