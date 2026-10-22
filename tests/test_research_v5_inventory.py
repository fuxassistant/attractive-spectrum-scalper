"""Coverage exclusions must not silently skip calendar gaps or contract rolls."""
from scripts.inventory_opportunity_v5 import previous_weekday, sufficient_context


def complete(symbol="GCZ6"):
    return dict(symbol=symbol, full_observed_window=True, current_audit_eligible=True)


def test_previous_weekday_keeps_missing_friday_instead_of_skipping_back():
    assert previous_weekday("2026-08-03") == "2026-07-31"
    assert previous_weekday("2026-09-01") == "2026-08-31"


def test_partial_previous_session_disqualifies_full_current_session():
    previous = complete()
    previous["full_observed_window"] = False
    assert not sufficient_context(complete(), previous)


def test_contract_roll_is_not_valid_previous_context():
    assert not sufficient_context(complete(), complete("GCQ6"))


def test_current_or_previous_overlay_ineligibility_is_respected():
    excluded = complete()
    excluded["current_audit_eligible"] = False
    assert not sufficient_context(excluded, complete())
    assert not sufficient_context(complete(), excluded)


def test_context_only_previous_day_does_not_recursively_require_its_own_previous():
    previous = complete()
    previous["own_previous_day_missing"] = True
    assert sufficient_context(complete(), previous)
