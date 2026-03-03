"""
Compare Python reference trackers against Cython implementations.

Loads detections from a shared JSONL file, optionally tiles them to increase
data size (scalability testing), and verifies that each Cython tracker
produces numerically identical results to its Python reference.

Also tests CLI (subprocess) and binary-CLI (PyInstaller) implementations
against the same reference, with relaxed tolerance for text-serialized output.
"""

import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import pytest


DETECTION_PATH = os.path.join(os.path.dirname(__file__), "data", "detection.jsonl")
CLI_TOLERANCE = 1e-6  # Relaxed for :.4f text formatting
IMG_H, IMG_W = 1080, 1920
SCALES = [1, 2, 4, 6, 8, 10]
LENGTHS = [1, 4]

TRACKER_TOLERANCE = {
    "SORT": None,
    "OC-SORT": None,
    "ByteTrack": 1e-6,
}

# On macOS Intel with Python 3.10, SORT can have tiny floating-point drift.
if (
    sys.platform == "darwin"
    and platform.machine() == "x86_64"
    and sys.version_info[:2] == (3, 10)
):
    TRACKER_TOLERANCE["SORT"] = 1e-16

_PERF_RECORDS: list[dict] = []


# ============================================================
# Shared helpers
# ============================================================


def load_detection_results(path: str) -> list[dict]:
    assert os.path.exists(path), f"Detection file not found: {path}"
    results = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    assert results, f"No detection results found in {path}"
    return results


def _parse_frame_detections(frame_result: dict) -> np.ndarray:
    """Extract Nx5 float64 detections from a frame dict."""
    detections = frame_result.get("detections", frame_result.get("bboxes", []))
    if len(detections) == 0:
        return np.empty((0, 5), dtype=np.float64)
    dets = np.array(detections, dtype=np.float64)
    if dets.size == 0:
        return np.empty((0, 5), dtype=np.float64)
    if dets.shape[1] < 5:
        scores = np.ones((dets.shape[0], 1), dtype=np.float64)
        dets = np.concatenate([dets, scores], axis=1)
    return dets[:, :5]


def stack_detections(
    detection_results: list[dict],
    scale: int,
    img_h: float = IMG_H,
    img_w: float = IMG_W,
) -> list[dict]:
    """Tile each frame's detections to create *scale*x the data.

    Layout (scale -> rows x cols):
      1x -> 1x1, 2x -> 2x1, 4x -> 2x2, 6x -> 2x3, 8x -> 2x4, 10x -> 2x5

    Each tile beyond (0, 0) offsets bboxes by ``(row * img_h, col * img_w)``.
    """
    if scale == 1:
        return detection_results

    rows = 2
    cols = scale // 2

    stacked: list[dict] = []
    for frame in detection_results:
        dets = _parse_frame_detections(frame)
        if dets.shape[0] == 0:
            stacked.append(frame)
            continue

        tiles = [dets]
        for r in range(rows):
            for c in range(cols):
                if r == 0 and c == 0:
                    continue
                tile = dets.copy()
                tile[:, 0] += c * img_w  # x1
                tile[:, 2] += c * img_w  # x2
                tile[:, 1] += r * img_h  # y1
                tile[:, 3] += r * img_h  # y2
                tiles.append(tile)

        merged = np.concatenate(tiles, axis=0)
        stacked.append(
            {
                "frame_idx": frame["frame_idx"],
                "detections": merged.tolist(),
            }
        )

    return stacked


def repeat_detections(detection_results: list[dict], length: int) -> list[dict]:
    """Repeat the detection sequence *length* times to simulate a longer video.

    Frame indices are offset so each repetition has unique indices.
    """
    if length == 1:
        return detection_results

    n_frames = len(detection_results)
    repeated: list[dict] = []
    for rep in range(length):
        for frame in detection_results:
            repeated.append(
                {
                    "frame_idx": frame["frame_idx"] + rep * n_frames,
                    "detections": frame.get("detections", frame.get("bboxes", [])),
                }
            )
    return repeated


def _avg_dets_per_frame(detection_results: list[dict]) -> float:
    """Compute average number of detections per frame."""
    total = sum(len(_parse_frame_detections(f)) for f in detection_results)
    return total / len(detection_results) if detection_results else 0.0


UpdateFn = Callable[[Any, np.ndarray], np.ndarray]


def run_tracker(
    tracker: Any,
    detection_results: list[dict],
    update_fn: UpdateFn,
) -> tuple[dict[int, npt.NDArray[np.floating]], dict[str, float]]:
    """Run *tracker* on *detection_results* using *update_fn* for the per-frame call."""
    tracking_results: dict[int, npt.NDArray[np.floating]] = {}
    frame_times: list[float] = []

    t0 = time.perf_counter()
    for frame_result in detection_results:
        frame_idx: int = frame_result["frame_idx"]
        dets = _parse_frame_detections(frame_result)
        t_frame = time.perf_counter()
        tracking_results[frame_idx] = update_fn(tracker, dets)
        frame_times.append(time.perf_counter() - t_frame)
    total = time.perf_counter() - t0

    if frame_times:
        arr = np.array(frame_times)
        perf = {
            "total_time": total,
            "num_frames": len(frame_times),
            "avg_ms": float(np.mean(arr)) * 1000,
            "median_ms": float(np.median(arr)) * 1000,
            "fps": len(frame_times) / total if total > 0 else 0.0,
        }
    else:
        perf = {
            "total_time": 0.0,
            "num_frames": 0,
            "avg_ms": 0.0,
            "median_ms": 0.0,
            "fps": 0.0,
        }

    return tracking_results, perf


# ============================================================
# CLI helpers
# ============================================================


def _format_dets_for_cli(dets: np.ndarray) -> str:
    """Format Nx5 detection array as CLI input line (x1,y1,x2,y2,score space-separated)."""
    if dets.size == 0:
        return ""
    parts = []
    for row in dets:
        parts.append(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]}")
    return " ".join(parts)


def _parse_cli_output_line(line: str) -> np.ndarray:
    """Parse CLI output line (x1,y1,x2,y2,id ...) into Nx5 array [x1,y1,x2,y2,id]."""
    line = line.strip()
    if not line:
        return np.empty((0, 5), dtype=np.float64)
    tracks = []
    for token in line.split():
        parts = token.split(",")
        x1, y1, x2, y2 = (
            float(parts[0]),
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
        )
        tid = float(parts[4])
        tracks.append([x1, y1, x2, y2, tid])
    return np.array(tracks, dtype=np.float64)


CLI_INIT_WAIT = 5  # Seconds to wait for CLI process initialization


def run_cli_tracker(
    cmd: list[str],
    detection_results: list[dict],
    timeout: int = 120,
) -> tuple[dict[int, npt.NDArray[np.floating]], dict[str, float]]:
    """Run a tracker via CLI subprocess, feeding detections through stdin.

    Returns (tracking_results, perf) with the same shape as run_tracker.
    """
    # Build input text: one line per frame
    input_lines = []
    frame_indices = []
    for frame_result in detection_results:
        frame_indices.append(frame_result["frame_idx"])
        dets = _parse_frame_detections(frame_result)
        input_lines.append(_format_dets_for_cli(dets))
    input_text = "\n".join(input_lines) + "\n"

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(CLI_INIT_WAIT)

    t0 = time.perf_counter()
    stdout, stderr = proc.communicate(input=input_text, timeout=timeout)
    total_time = time.perf_counter() - t0

    if proc.returncode != 0:
        stderr_truncated = stderr[:2000] if stderr else "(no stderr)"
        raise RuntimeError(
            f"CLI process failed: cmd={cmd}, exit_code={proc.returncode}\n"
            f"stderr: {stderr_truncated}"
        )

    # Parse output lines
    output_text = stdout
    if output_text.endswith("\n"):
        output_text = output_text[:-1]
    output_lines = output_text.split("\n") if output_text else []

    if len(output_lines) != len(frame_indices):
        raise RuntimeError(
            f"CLI output line count mismatch: expected {len(frame_indices)}, "
            f"got {len(output_lines)} (cmd={cmd})"
        )

    tracking_results: dict[int, npt.NDArray[np.floating]] = {}
    for frame_idx, line in zip(frame_indices, output_lines):
        tracking_results[frame_idx] = _parse_cli_output_line(line)

    num_frames = len(frame_indices)
    perf = {
        "total_time": total_time,
        "num_frames": num_frames,
        "avg_ms": (total_time / num_frames * 1000) if num_frames > 0 else 0.0,
        "median_ms": 0.0,  # No per-frame granularity through a pipe
        "fps": num_frames / total_time if total_time > 0 else 0.0,
    }

    return tracking_results, perf


# ============================================================
# Perf recording
# ============================================================


def _record_perf(
    tracker: str,
    impl: str,
    scale: int,
    length: int,
    avg_dets: float,
    perf: dict[str, float],
) -> None:
    """Append a tidy perf record to _PERF_RECORDS."""
    _PERF_RECORDS.append(
        {
            "tracker": tracker,
            "impl": impl,
            "scale": scale,
            "length": length,
            "avg_dets": avg_dets,
            "total_time": perf["total_time"],
            "num_frames": perf["num_frames"],
            "avg_ms": perf["avg_ms"],
            "median_ms": perf["median_ms"],
            "fps": perf["fps"],
        }
    )


# ============================================================
# Comparison helpers
# ============================================================


def _compare_frames(a: np.ndarray, b: np.ndarray, tolerance: float | None) -> bool:
    if len(a) != len(b):
        return False
    if len(a) == 0:
        return True
    # Sort by track ID for robustness
    a = a[a[:, 4].argsort()]
    b = b[b[:, 4].argsort()]
    if not np.array_equal(a[:, 4], b[:, 4]):
        return False
    if tolerance is None:
        return bool(np.array_equal(a[:, :4], b[:, :4]))
    return bool(np.allclose(a[:, :4], b[:, :4], atol=tolerance, rtol=0))


def assert_results_match(
    results_ref: dict[int, npt.NDArray[np.floating]],
    results_test: dict[int, npt.NDArray[np.floating]],
    tracker_name: str,
    tolerance: float | None,
    perf_ref: dict[str, float] | None = None,
    perf_test: dict[str, float] | None = None,
) -> None:
    all_frames = sorted(set(results_ref) | set(results_test))
    empty = np.empty((0, 5), dtype=np.float64)
    n_differ = 0
    first_diff_frame: int | None = None

    for frame_idx in all_frames:
        a = results_ref.get(frame_idx, empty)
        b = results_test.get(frame_idx, empty)
        if not _compare_frames(a, b, tolerance):
            n_differ += 1
            if first_diff_frame is None:
                first_diff_frame = frame_idx

    if perf_ref and perf_test:
        speedup = (
            perf_ref["total_time"] / perf_test["total_time"]
            if perf_test["total_time"] > 0
            else 0
        )
        print(
            f"\n{tracker_name}: {len(all_frames)} frames | "
            f"Ref {perf_ref['avg_ms']:.3f}ms/frame ({perf_ref['fps']:.0f} fps) | "
            f"Test {perf_test['avg_ms']:.3f}ms/frame ({perf_test['fps']:.0f} fps) | "
            f"{speedup:.1f}x speedup"
        )

    if n_differ > 0:
        pytest.fail(
            f"{tracker_name}: {n_differ}/{len(all_frames)} frames differ "
            f"(first at frame {first_diff_frame})"
        )


# ============================================================
# Test functions
# ============================================================


@pytest.mark.parametrize("length", LENGTHS, ids=[f"len{l}" for l in LENGTHS])
@pytest.mark.parametrize("scale", SCALES, ids=[f"s{s}" for s in SCALES])
def test_sort_comparison(scale: int, length: int, request):
    detection_results = load_detection_results(DETECTION_PATH)
    detection_results = stack_detections(detection_results, scale)
    detection_results = repeat_detections(detection_results, length)
    avg_dets = _avg_dets_per_frame(detection_results)
    label = f"SORT s{scale}-len{length}"

    # 1. Python reference
    from references.sort.sort import Sort as SortPython, KalmanBoxTracker
    from pyxtrackers.sort.sort import Sort as SortCython

    def update(tracker, dets):
        return tracker.update(dets)

    KalmanBoxTracker.count = 0
    results_py, perf_py = run_tracker(
        SortPython(max_age=20, min_hits=1, iou_threshold=0.1),
        detection_results,
        update,
    )
    _record_perf("SORT", "python", scale, length, avg_dets, perf_py)

    # 2. Cython direct
    KalmanBoxTracker.count = 0
    results_cy, perf_cy = run_tracker(
        SortCython(max_age=20, min_hits=1, iou_threshold=0.1),
        detection_results,
        update,
    )
    assert_results_match(
        results_py,
        results_cy,
        f"{label} [cython]",
        tolerance=TRACKER_TOLERANCE["SORT"],
        perf_ref=perf_py,
        perf_test=perf_cy,
    )
    _record_perf("SORT", "cython", scale, length, avg_dets, perf_cy)

    # 3. CLI (always runs)
    cmd = [
        sys.executable,
        "-m",
        "pyxtrackers.cli_launcher",
        "sort",
        "--max-age",
        "20",
        "--min-hits",
        "1",
        "--iou-threshold",
        "0.1",
    ]
    results_cli, perf_cli = run_cli_tracker(cmd, detection_results)
    assert_results_match(
        results_py,
        results_cli,
        f"{label} [cli]",
        tolerance=CLI_TOLERANCE,
        perf_ref=perf_py,
        perf_test=perf_cli,
    )
    _record_perf("SORT", "cli", scale, length, avg_dets, perf_cli)

    # 4. Binary CLI (only if --cli-binary provided)
    binary_path = request.config.getoption("--cli-binary")
    if binary_path:
        cmd = [
            binary_path,
            "sort",
            "--max-age",
            "20",
            "--min-hits",
            "1",
            "--iou-threshold",
            "0.1",
        ]
        results_bin, perf_bin = run_cli_tracker(cmd, detection_results)
        assert_results_match(
            results_py,
            results_bin,
            f"{label} [binary-cli]",
            tolerance=CLI_TOLERANCE,
            perf_ref=perf_py,
            perf_test=perf_bin,
        )
        _record_perf("SORT", "binary-cli", scale, length, avg_dets, perf_bin)


@pytest.mark.parametrize("length", LENGTHS, ids=[f"len{l}" for l in LENGTHS])
@pytest.mark.parametrize("scale", SCALES, ids=[f"s{s}" for s in SCALES])
def test_ocsort_comparison(scale: int, length: int, request):
    detection_results = load_detection_results(DETECTION_PATH)
    detection_results = stack_detections(detection_results, scale)
    detection_results = repeat_detections(detection_results, length)
    avg_dets = _avg_dets_per_frame(detection_results)
    label = f"OC-SORT s{scale}-len{length}"

    from references.ocsort.ocsort import OCSort as OCSortPython, KalmanBoxTracker
    from pyxtrackers.ocsort.ocsort import OCSort as OCSortCython

    img_h = IMG_H * (2 if scale > 1 else 1)
    img_w = IMG_W * max(scale // 2, 1)
    img_info, img_size = (img_h, img_w), (img_h, img_w)

    def update_python(tracker, dets):
        return tracker.update(dets, img_info, img_size)

    def update_cython(tracker, dets):
        return tracker.update(dets)

    # 1. Python reference
    KalmanBoxTracker.count = 0
    results_py, perf_py = run_tracker(
        OCSortPython(det_thresh=0.3),
        detection_results,
        update_python,
    )
    _record_perf("OC-SORT", "python", scale, length, avg_dets, perf_py)

    # 2. Cython direct
    KalmanBoxTracker.count = 0
    results_cy, perf_cy = run_tracker(
        OCSortCython(det_thresh=0.3),
        detection_results,
        update_cython,
    )
    assert_results_match(
        results_py,
        results_cy,
        f"{label} [cython]",
        tolerance=TRACKER_TOLERANCE["OC-SORT"],
        perf_ref=perf_py,
        perf_test=perf_cy,
    )
    _record_perf("OC-SORT", "cython", scale, length, avg_dets, perf_cy)

    # 3. CLI (always runs)
    cmd = [
        sys.executable,
        "-m",
        "pyxtrackers.cli_launcher",
        "ocsort",
        "--det-thresh",
        "0.3",
    ]
    results_cli, perf_cli = run_cli_tracker(cmd, detection_results)
    assert_results_match(
        results_py,
        results_cli,
        f"{label} [cli]",
        tolerance=CLI_TOLERANCE,
        perf_ref=perf_py,
        perf_test=perf_cli,
    )
    _record_perf("OC-SORT", "cli", scale, length, avg_dets, perf_cli)

    # 4. Binary CLI (only if --cli-binary provided)
    binary_path = request.config.getoption("--cli-binary")
    if binary_path:
        cmd = [binary_path, "ocsort", "--det-thresh", "0.3"]
        results_bin, perf_bin = run_cli_tracker(cmd, detection_results)
        assert_results_match(
            results_py,
            results_bin,
            f"{label} [binary-cli]",
            tolerance=CLI_TOLERANCE,
            perf_ref=perf_py,
            perf_test=perf_bin,
        )
        _record_perf("OC-SORT", "binary-cli", scale, length, avg_dets, perf_bin)


@pytest.mark.parametrize("length", LENGTHS, ids=[f"len{l}" for l in LENGTHS])
@pytest.mark.parametrize("scale", SCALES, ids=[f"s{s}" for s in SCALES])
def test_bytetrack_comparison(scale: int, length: int, request):
    detection_results = load_detection_results(DETECTION_PATH)
    detection_results = stack_detections(detection_results, scale)
    detection_results = repeat_detections(detection_results, length)
    avg_dets = _avg_dets_per_frame(detection_results)
    label = f"ByteTrack s{scale}-len{length}"

    from references.bytetrack.byte_tracker import BYTETracker as BYTETrackerPython
    from references.bytetrack.basetrack import BaseTrack
    from pyxtrackers.bytetrack.bytetrack import BYTETracker as BYTETrackerCython

    img_h = IMG_H * (2 if scale > 1 else 1)
    img_w = IMG_W * max(scale // 2, 1)
    img_info, img_size = (img_h, img_w), (img_h, img_w)

    def update_python(tracker, dets):
        tracked_objs = tracker.update(dets, img_info, img_size)
        if isinstance(tracked_objs, np.ndarray):
            return tracked_objs
        if len(tracked_objs) > 0:
            return np.array(
                [
                    [t.tlbr[0], t.tlbr[1], t.tlbr[2], t.tlbr[3], t.track_id]
                    for t in tracked_objs
                ],
                dtype=np.float64,
            )
        return np.empty((0, 5), dtype=np.float64)

    def update_cython(tracker, dets):
        return tracker.update(dets)

    class Args:
        track_thresh = 0.5
        track_buffer = 30
        match_thresh = 0.8
        mot20 = False

    # 1. Python reference
    BaseTrack._count = 0
    results_py, perf_py = run_tracker(
        BYTETrackerPython(Args()),
        detection_results,
        update_python,
    )
    _record_perf("ByteTrack", "python", scale, length, avg_dets, perf_py)

    # 2. Cython direct
    BaseTrack._count = 0
    results_cy, perf_cy = run_tracker(
        BYTETrackerCython(
            track_thresh=0.5, match_thresh=0.8, track_buffer=30, mot20=False
        ),
        detection_results,
        update_cython,
    )
    assert_results_match(
        results_py,
        results_cy,
        f"{label} [cython]",
        tolerance=TRACKER_TOLERANCE["ByteTrack"],
        perf_ref=perf_py,
        perf_test=perf_cy,
    )
    _record_perf("ByteTrack", "cython", scale, length, avg_dets, perf_cy)

    # 3. CLI (always runs)
    cmd = [
        sys.executable,
        "-m",
        "pyxtrackers.cli_launcher",
        "bytetrack",
        "--track-thresh",
        "0.5",
        "--match-thresh",
        "0.8",
        "--track-buffer",
        "30",
    ]
    results_cli, perf_cli = run_cli_tracker(cmd, detection_results)
    assert_results_match(
        results_py,
        results_cli,
        f"{label} [cli]",
        tolerance=CLI_TOLERANCE,
        perf_ref=perf_py,
        perf_test=perf_cli,
    )
    _record_perf("ByteTrack", "cli", scale, length, avg_dets, perf_cli)

    # 4. Binary CLI (only if --cli-binary provided)
    binary_path = request.config.getoption("--cli-binary")
    if binary_path:
        cmd = [
            binary_path,
            "bytetrack",
            "--track-thresh",
            "0.5",
            "--match-thresh",
            "0.8",
            "--track-buffer",
            "30",
        ]
        results_bin, perf_bin = run_cli_tracker(cmd, detection_results)
        assert_results_match(
            results_py,
            results_bin,
            f"{label} [binary-cli]",
            tolerance=CLI_TOLERANCE,
            perf_ref=perf_py,
            perf_test=perf_bin,
        )
        _record_perf("ByteTrack", "binary-cli", scale, length, avg_dets, perf_bin)
