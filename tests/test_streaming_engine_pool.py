from miles.rollout.streaming_rollout_manager import EnginePool


def test_drain_policy_keeps_one_at_a_time_updates():
    pool = EnginePool(
        ["http://e0", "http://e1", "http://e2", "http://e3"],
        initial_version=0,
        min_active_engines=3,
        rolling_updates_enabled=True,
    )
    pool.notify_new_version(1)

    summary = pool.summary()
    assert summary["num_active"] >= 3
    assert summary["num_drain_only"] == 1


def test_get_update_candidates_returns_at_most_one():
    pool = EnginePool(
        ["http://e0", "http://e1", "http://e2"],
        initial_version=0,
        min_active_engines=1,
        rolling_updates_enabled=True,
    )
    pool.notify_new_version(1)

    # With min_active_engines=1, multiple outdated engines can be marked drain-only.
    drained = [e.engine_idx for e in pool.engines if e.drain_only]
    assert len(drained) >= 1

    # Ensure they are idle so they'd all be eligible.
    for idx in drained:
        pool.engines[idx].inflight_groups = 0

    candidates = pool.get_update_candidates()
    assert len(candidates) <= 1


def test_selection_prefers_non_draining_engines():
    pool = EnginePool(
        ["http://e0", "http://e1"],
        initial_version=0,
        min_active_engines=1,
        rolling_updates_enabled=True,
    )
    pool.engines[0].drain_only = True
    pool.engines[1].drain_only = False

    chosen = pool.select_engine()
    assert chosen is not None
    assert chosen.engine_idx == 1


def test_engine_quarantine_after_three_failures():
    pool = EnginePool(
        ["http://e0", "http://e1"],
        initial_version=0,
        min_active_engines=1,
        rolling_updates_enabled=True,
    )

    pool.mark_failure(0)
    pool.mark_failure(0)
    pool.mark_failure(0)
    assert pool.engines[0].healthy is False

    chosen = pool.select_engine()
    assert chosen is not None
    assert chosen.engine_idx == 1

    # Mark healthy again and ensure it can be selected.
    pool.mark_success(0)
    pool.engines[0].drain_only = False
    chosen2 = pool.select_engine()
    assert chosen2 is not None
    assert chosen2.engine_idx in [0, 1]


def test_single_engine_disables_rolling_updates_but_allows_selection():
    pool = EnginePool(
        ["http://e0"],
        initial_version=0,
        min_active_engines=1,
        rolling_updates_enabled=False,
    )

    pool.notify_new_version(1)
    assert pool.get_update_candidates() == []

    chosen = pool.select_engine()
    assert chosen is not None
    assert chosen.engine_idx == 0


if __name__ == "__main__":
    test_drain_policy_keeps_one_at_a_time_updates()
    test_get_update_candidates_returns_at_most_one()
    test_selection_prefers_non_draining_engines()
    test_engine_quarantine_after_three_failures()
    test_single_engine_disables_rolling_updates_but_allows_selection()
