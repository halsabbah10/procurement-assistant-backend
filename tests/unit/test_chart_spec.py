from app.agent.chart_spec import ChartDecision, _flatten, _is_numeric, _sanitize_decision


def test_flatten_leaves_scalars_untouched():
    assert _flatten("Health Care Services, Department of") == "Health Care Services, Department of"
    assert _flatten(42) == 42


def test_flatten_joins_compound_group_key_values():
    assert _flatten({"fiscal_year": "2013-2014", "quarter": 3}) == "2013-2014 3"


def test_is_numeric_rejects_bool_despite_being_an_int_subclass():
    # Python's bool is a subclass of int — True/False must never pass as a
    # legitimate chart value just because isinstance(True, int) is True.
    assert _is_numeric(True) is False
    assert _is_numeric(1) is True
    assert _is_numeric(1.5) is True
    assert _is_numeric("1") is False


def test_sanitize_decision_rejects_none_type():
    decision = ChartDecision(chart_type="none", category_field="_id", value_field="total")
    rows = [{"_id": "A", "total": 1}, {"_id": "B", "total": 2}]
    assert _sanitize_decision(decision, rows) is None


def test_sanitize_decision_rejects_single_row():
    decision = ChartDecision(chart_type="bar", category_field="_id", value_field="total")
    rows = [{"_id": "A", "total": 1}]
    assert _sanitize_decision(decision, rows) is None


def test_sanitize_decision_rejects_missing_fields():
    decision = ChartDecision(chart_type="bar", category_field="dept", value_field="total")
    rows = [{"_id": "A", "total": 1}, {"_id": "B", "total": 2}]
    assert _sanitize_decision(decision, rows) is None


def test_sanitize_decision_rejects_non_numeric_value_field():
    decision = ChartDecision(chart_type="bar", category_field="_id", value_field="total")
    rows = [{"_id": "A", "total": "not a number"}, {"_id": "B", "total": 2}]
    assert _sanitize_decision(decision, rows) is None


def test_sanitize_decision_accepts_valid_bar_chart():
    decision = ChartDecision(
        chart_type="bar", category_field="_id", value_field="total_spending", title="Top depts"
    )
    rows = [
        {"_id": "Health Care Services", "total_spending": 100.0},
        {"_id": "Public Health", "total_spending": 50.0},
    ]
    result = _sanitize_decision(decision, rows)
    assert result == {
        "type": "bar",
        "title": "Top depts",
        "category_field": "_id",
        "value_field": "total_spending",
        "data": [
            {"category": "Health Care Services", "value": 100.0},
            {"category": "Public Health", "value": 50.0},
        ],
    }


def test_sanitize_decision_trims_to_max_categories():
    decision = ChartDecision(chart_type="bar", category_field="_id", value_field="total")
    rows = [{"_id": str(i), "total": i} for i in range(20)]
    result = _sanitize_decision(decision, rows)
    assert len(result["data"]) == 12
