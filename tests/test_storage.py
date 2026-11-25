from src.storage import delete_evaluation, get_evaluation, list_evaluations, save_evaluation


def test_sqlite_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("EVALUATIONS_DB", str(tmp_path / "evaluations.db"))
    report = {
        "mode": "idea",
        "title": "Test idea",
        "language": "en",
        "preset": "tunisia",
        "overall_score": 71.2,
        "success_probability": 64.0,
        "financials": {"roi_percent": 42.0},
    }
    report_id = save_evaluation(report)
    assert report_id > 0
    assert report["report_id"] == report_id

    records = list_evaluations(limit=10)
    assert len(records) == 1
    assert records[0]["title"] == "Test idea"

    loaded = get_evaluation(report_id)
    assert loaded is not None
    assert loaded["overall_score"] == 71.2
    assert loaded["report_id"] == report_id

    assert delete_evaluation(report_id) is True
    assert get_evaluation(report_id) is None
