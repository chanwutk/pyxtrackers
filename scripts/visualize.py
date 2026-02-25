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

IMPL_MAPPING = [
    ("cython", "Cython (Direct Import)"),
    ("cli", "Cython (CLI)"),
    # ("binary-cli", "Cython (Binary CLI)"),
    ("python", "Python"),
]


def get_implementation(impl: str) -> str:
    return dict(IMPL_MAPPING).get(impl, impl)


def generate_charts(records: list[dict], output_dir: str) -> None:
    import altair as alt

    os.makedirs(output_dir, exist_ok=True)

    tracker_shape = alt.Shape(
        "tracker:N",
        scale=alt.Scale(
            domain=["SORT", "ByteTrack", "OC-SORT"],
            range=["triangle", "circle", "diamond", "square"],
        ),
        legend=alt.Legend(orient="bottom", title="Tracker"),
    )
    python_label = get_implementation("python")

    def build_line_points(rows: list[dict], impls: list[str]) -> alt.LayerChart:
        dash_range = [[4, 4] if i == python_label else [0, 0] for i in impls]
        opacity_range = [0.4 if i == python_label else 1.0 for i in impls]

        base = alt.Chart(alt.Data(values=rows)).encode(shape=tracker_shape)
        line = base.mark_line().encode(
            color=alt.Color(
                "impl:N", legend=alt.Legend(orient="bottom", title="Implementation")
            ),
            strokeDash=alt.StrokeDash(
                "impl:N",
                scale=alt.Scale(domain=impls, range=dash_range),
                legend=None,
            ),
            opacity=alt.Opacity(
                "impl:N",
                scale=alt.Scale(domain=impls, range=opacity_range),
                legend=None,
            ),
        )
        points = base.mark_point(size=150, opacity=1, strokeWidth=2).encode(
            fill=alt.value("white"),
            stroke=alt.Stroke("impl:N", legend=None),
            opacity=alt.Opacity(
                "impl:N",
                scale=alt.Scale(domain=impls, range=opacity_range),
                legend=None,
            ),
        )
        return line + points

    # -- Plot 1: Throughput (fps) -------------------------------------------
    throughput_rows = [
        {
            "tracker": r["tracker"],
            "avg_dets": r["avg_dets"],
            "fps": r["fps"],
            "impl": get_implementation(r["impl"]),
        }
        for r in records
        if r["impl"] != "binary-cli"
    ]
    # Identify which implementations are present
    impls = [i[1] for i in IMPL_MAPPING]
    non_python_impls = [i for i in impls if i != python_label]

    base_throughput = build_line_points(throughput_rows, impls).encode(
        x=alt.X("avg_dets:Q", title="Avg detections per frame"),
        detail="tracker:N",
        tooltip=["tracker:N", "impl:N", "avg_dets:Q", "fps:Q"],
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
        if r["impl"] == "python" or r["impl"] == "binary-cli":
            continue
        key = (r["tracker"], r["scale"], r["length"])
        py_time = python_times.get(key)
        if py_time and r["total_time"] > 0:
            speedup_rows.append(
                {
                    "tracker": r["tracker"],
                    "avg_dets": r["avg_dets"],
                    "impl": get_implementation(r["impl"]),
                    "speedup": py_time / r["total_time"],
                }
            )

    if not speedup_rows:
        print("No speedup data — skipping speedup chart.")
        return

    chart2 = (
        build_line_points(speedup_rows, impls)
        .encode(
            x=alt.X("avg_dets:Q", title="Avg detections per frame"),
            y=alt.Y("speedup:Q", title="Speedup vs Python"),
            detail="tracker:N",
            tooltip=["tracker:N", "impl:N", "avg_dets:Q", "speedup:Q"],
        )
        .properties(
            title="Speedup over Python by implementation", width=860, height=400
        )
    )

    speedup_path = os.path.join(output_dir, "speedup.png")
    chart2.save(speedup_path, scale_factor=2)
    print(f"Saved: {speedup_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate benchmark charts from perf.jsonl"
    )
    parser.add_argument(
        "--input",
        default="test-results/perf.jsonl",
        help="Path to perf.jsonl (default: test-results/perf.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        default="assets",
        help="Output directory for charts (default: assets/)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(
            f"Error: {args.input} not found. Run pytest first to generate perf data.",
            file=sys.stderr,
        )
        sys.exit(1)

    records = load_records(args.input)
    if not records:
        print(f"Error: {args.input} is empty.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(records)} records from {args.input}")
    generate_charts(records, args.output_dir)


if __name__ == "__main__":
    main()
