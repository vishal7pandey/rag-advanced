from app.core.memory.window import WindowMemory


def test_add_get_and_limits():
    win = WindowMemory()
    assert win.get(5) == []
    win.add("user", "q1")
    win.add("assistant", "a1")
    win.add("user", "q2")
    last_two = win.get(2)
    assert len(last_two) == 2
    assert last_two[0].content == "a1"
    assert last_two[1].content == "q2"
    # hydration via property
    turns = win.turns
    assert len(turns) == 3
    assert turns[0].role == "user"
