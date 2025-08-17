from app.core.answer_guard import should_no_answer, caveat_text


def test_no_answer_when_no_retrieved():
    assert should_no_answer(0, None) is True


def test_no_answer_when_low_precision_and_single_source():
    assert should_no_answer(1, 0.2) is True


def test_answer_ok_with_multiple_sources_and_ok_precision():
    assert should_no_answer(3, 0.85) is False


def test_caveat_triggers_on_low_groundedness():
    assert caveat_text({"groundedness_lite": 0.6}) is not None


def test_caveat_none_when_metrics_good():
    assert caveat_text({"context_precision_lite": 0.95, "groundedness_lite": 0.9}) is None
