from src.agent import FIELD_ORDER, start_conversation, process_message


def test_state_machine_collects_and_parses_core_fields():
    state, welcome = start_conversation("en", "tunisia", "idea")
    assert state.stage == "description"
    assert "CEO AI" in welcome or "evaluate" in welcome.lower()
    turn = process_message(state, "I want to launch a SaaS for restaurants.")
    assert state.stage == "fields"
    assert turn.current_field == "sector"
    process_message(state, "software / SaaS platform")
    assert state.fields["sector"] == "saas"
    process_message(state, "B2B restaurants")
    assert state.fields["target_market"] == "b2b"
    process_message(state, "Tunis")
    assert state.fields["location"] == "Tunis"
    process_message(state, "45000")
    assert state.fields["budget"] == 45000
    process_message(state, "4")
    assert int(state.fields["team_size"]) == 4


def test_skip_uses_defaults():
    state, _ = start_conversation("fr", "tunisia", "idea")
    process_message(state, "Une application de livraison pour les petits commerces.")
    process_message(state, "ecommerce")
    process_message(state, "passer")
    assert state.fields["target_market"] == "b2c"
    assert state.default_used


def test_skip_every_field_reaches_a_report():
    for language in ("en", "fr", "ar"):
        state, _ = start_conversation(language, "tunisia", "idea")
        process_message(state, "A small local business.")
        turn = None
        for _ in range(len(FIELD_ORDER) + 2):
            turn = process_message(state, "skip")
            if turn.ready:
                break
        assert turn is not None and turn.ready, language
        assert turn.result is not None, language
        assert turn.result["created_inputs"]["sector"] == "other", language


def test_evaluate_now_fills_missing_fields():
    state, _ = start_conversation("en", "tunisia", "idea")
    process_message(state, "A subscription platform for independent restaurants.")
    turn = process_message(state, "evaluate now")
    assert turn.ready is True
    assert turn.result is not None
    assert state.stage == "complete"
    assert all(field in turn.result["created_inputs"] for field in FIELD_ORDER)
