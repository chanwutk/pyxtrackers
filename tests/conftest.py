"""Pytest configuration for pyxtrackers tests.

Adds a ``--visualize`` flag that switches between:
- **Normal mode** (default): only ``len1`` tests run (fast correctness check).
- **Visualization mode**: all lengths run, and after the session finishes two
  benchmark plots are saved to ``assets/``.
"""

import os


def pytest_addoption(parser):
    parser.addoption(
        "--visualize",
        action="store_true",
        default=False,
        help="Run full scalability benchmarks and generate plots in assets/",
    )


# def pytest_collection_modifyitems(config, items):
#     if config.getoption("--visualize"):
#         return
#     deselected = []
#     selected = []
#     for item in items:
#         if any(f"len{l}" in item.nodeid for l in (4,)):
#             deselected.append(item)
#         else:
#             selected.append(item)
#     if deselected:
#         config.hook.pytest_deselected(items=deselected)
#         items[:] = selected


def pytest_sessionfinish(session, exitstatus):
    if not session.config.getoption("--visualize", default=False):
        return

    from tests.test_trackers import _PERF_RECORDS

    import altair as alt

    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    os.makedirs(assets_dir, exist_ok=True)

    # -- Plot 1: Throughput (fps) -------------------------------------------
    # Reshape into tidy (long) format: one row per (tracker, impl, frame_count)
    throughput_rows = []
    for r in _PERF_RECORDS:
        throughput_rows.append(
            {"tracker": r["tracker"], "avg_dets": r["avg_dets"], "fps": r["fps_py"], "impl": "Python"}
        )
        throughput_rows.append(
            {"tracker": r["tracker"], "avg_dets": r["avg_dets"], "fps": r["fps_cy"], "impl": "Cython"}
        )

    base_throughput = (
        alt.Chart(alt.Data(values=throughput_rows))
        .mark_line(point=True)
        .encode(
            x=alt.X("avg_dets:Q", title="Avg detections per frame"),
            color=alt.Color("tracker:N", legend=alt.Legend(orient="bottom", title="Tracker")),
            strokeDash=alt.StrokeDash(
                "impl:N",
                scale=alt.Scale(domain=["Python", "Cython"], range=[[4, 4], [0, 0]]),
            ),
            opacity=alt.Opacity(
                "impl:N",
                scale=alt.Scale(domain=["Python", "Cython"], range=[0.4, 1.0]),
                legend=alt.Legend(orient="bottom", title="Implementation"),
            ),
            tooltip=["tracker:N", "impl:N", "avg_dets:Q", "fps:Q"],
        )
    )

    throughput_left = base_throughput.encode(
        y=alt.Y("fps:Q", title="Throughput (fps)")
    ).properties(title="Linear throughput scale", width=400, height=400)

    throughput_right = base_throughput.encode(
        y=alt.Y("fps:Q", title="Throughput (fps)", scale=alt.Scale(type="log"))
    ).properties(title="Log throughput scale", width=400, height=400)

    chart1 = (throughput_left | throughput_right).properties(
        title="Tracker throughput: Python (Original) vs Cython (Ours)"
    )

    throughput_path = os.path.join(assets_dir, "throughput.png")
    chart1.save(throughput_path, scale_factor=2)
    print(f"\nSaved: {throughput_path}")

    # -- Plot 2: Speedup ----------------------------------------------------
    speedup_rows = [
        {"tracker": r["tracker"], "avg_dets": r["avg_dets"], "speedup": r["speedup"]}
        for r in _PERF_RECORDS
    ]

    base_speedup = (
        alt.Chart(alt.Data(values=speedup_rows))
        .mark_line(point=True)
        .encode(
            x=alt.X("avg_dets:Q", title="Avg detections per frame"),
            color=alt.Color("tracker:N", legend=alt.Legend(orient="bottom")),
            tooltip=["tracker:N", "avg_dets:Q", "speedup:Q"],
        )
    )

    chart2 = (
        base_speedup.encode(y=alt.Y("speedup:Q", title="Speedup (Python time / Cython time)"))
        .properties(title="Cython Speedup over Python (Ours / Original)", width=860, height=400)
    )

    speedup_path = os.path.join(assets_dir, "speedup.png")
    chart2.save(speedup_path, scale_factor=2)
    print(f"Saved: {speedup_path}")
