"""Generate benchmark charts from perf.jsonl test results.

Usage:
    python scripts/visualize.py                              # defaults
    python scripts/visualize.py --input path/to/perf.jsonl   # custom input
    python scripts/visualize.py --output-dir custom/          # custom output dir
"""

import argparse
import json
import os
import sys


def load_records(path: str) -> list[dict]:
    records = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def generate_charts(records: list[dict], output_dir: str) -> None:
    import altair as alt

    os.makedirs(output_dir, exist_ok=True)

    # Identify which implementations are present
    impls = sorted(set(r["impl"] for r in records))
    non_python_impls = [i for i in impls if i != "python"]

    # -- Plot 1: Throughput (fps) -------------------------------------------
    throughput_rows = [
        {
            "tracker": r["tracker"],
            "avg_dets": r["avg_dets"],
            "fps": r["fps"],
            "impl": r["impl"],
        }
        for r in records
    ]

    # Python dashed/faded, all others solid/opaque
    dash_domain = impls
    dash_range = [[4, 4] if i == "python" else [0, 0] for i in impls]
    opacity_domain = impls
    opacity_range = [0.4 if i == "python" else 1.0 for i in impls]

    base_throughput = (
        alt.Chart(alt.Data(values=throughput_rows))
        .mark_line(point=True)
        .encode(
            x=alt.X("avg_dets:Q", title="Avg detections per frame"),
            color=alt.Color("tracker:N", legend=alt.Legend(orient="bottom", title="Tracker")),
            strokeDash=alt.StrokeDash(
                "impl:N",
                scale=alt.Scale(domain=dash_domain, range=dash_range),
            ),
            opacity=alt.Opacity(
                "impl:N",
                scale=alt.Scale(domain=opacity_domain, range=opacity_range),
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
        title="Tracker throughput by implementation"
    )

    throughput_path = os.path.join(output_dir, "throughput.png")
    chart1.save(throughput_path, scale_factor=2)
    print(f"Saved: {throughput_path}")

    # -- Plot 2: Speedup ----------------------------------------------------
    if not non_python_impls:
        print("No non-python implementations found — skipping speedup chart.")
        return

    # Build a lookup: (tracker, scale, length) -> python total_time
    python_times: dict[tuple, float] = {}
    for r in records:
        if r["impl"] == "python":
            key = (r["tracker"], r["scale"], r["length"])
            python_times[key] = r["total_time"]

    speedup_rows = []
    for r in records:
        if r["impl"] == "python":
            continue
        key = (r["tracker"], r["scale"], r["length"])
        py_time = python_times.get(key)
        if py_time and r["total_time"] > 0:
            speedup_rows.append({
                "tracker": r["tracker"],
                "avg_dets": r["avg_dets"],
                "impl": r["impl"],
                "speedup": py_time / r["total_time"],
            })

    if not speedup_rows:
        print("No speedup data — skipping speedup chart.")
        return

    # Distinguish impl via strokeDash
    su_dash_domain = non_python_impls
    su_dash_range = [[0, 0]] + [[4, 2]] * (len(non_python_impls) - 1) if len(non_python_impls) > 1 else [[0, 0]]

    base_speedup = (
        alt.Chart(alt.Data(values=speedup_rows))
        .mark_line(point=True)
        .encode(
            x=alt.X("avg_dets:Q", title="Avg detections per frame"),
            color=alt.Color("tracker:N", legend=alt.Legend(orient="bottom")),
            strokeDash=alt.StrokeDash(
                "impl:N",
                scale=alt.Scale(domain=su_dash_domain, range=su_dash_range),
                legend=alt.Legend(orient="bottom", title="Implementation"),
            ),
            tooltip=["tracker:N", "impl:N", "avg_dets:Q", "speedup:Q"],
        )
    )

    chart2 = (
        base_speedup.encode(y=alt.Y("speedup:Q", title="Speedup vs Python"))
        .properties(title="Speedup over Python by implementation", width=860, height=400)
    )

    speedup_path = os.path.join(output_dir, "speedup.png")
    chart2.save(speedup_path, scale_factor=2)
    print(f"Saved: {speedup_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark charts from perf.jsonl")
    parser.add_argument(
        "--input", default="test-results/perf.jsonl",
        help="Path to perf.jsonl (default: test-results/perf.jsonl)",
    )
    parser.add_argument(
        "--output-dir", default="assets",
        help="Output directory for charts (default: assets/)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found. Run pytest first to generate perf data.", file=sys.stderr)
        sys.exit(1)

    records = load_records(args.input)
    if not records:
        print(f"Error: {args.input} is empty.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(records)} records from {args.input}")
    generate_charts(records, args.output_dir)


if __name__ == "__main__":
    main()
