"""
Compare Python reference trackers against Cython implementations.

Loads detections from a shared JSONL file and verifies that each Cython
tracker produces numerically identical results to its Python reference.
"""

import json
import os
import time
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import pytest


DETECTION_PATH = os.path.join(os.path.dirname(__file__), "data", "detection.jsonl")
TOLERANCE = 1e-6


# ============================================================
# Shared helpers
# ============================================================

def load_detection_results(path: str) -> list[dict]:
    if not os.path.exists(path):
        pytest.skip(f"Detection file not found: {path}")
    results = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    if not results:
        pytest.skip(f"No detection results found in {path}")
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
        perf = {"total_time": 0.0, "num_frames": 0, "avg_ms": 0.0, "median_ms": 0.0, "fps": 0.0}

    return tracking_results, perf


def _compare_frames(
    result_a: np.ndarray, result_b: np.ndarray, tolerance: float
) -> bool:
    if len(result_a) != len(result_b):
        return False
    if len(result_a) == 0:
        return True
    a = result_a[result_a[:, 4].argsort()]
    b = result_b[result_b[:, 4].argsort()]
    if not np.array_equal(a[:, 4], b[:, 4]):
        return False
    return bool(np.allclose(a[:, :4], b[:, :4], atol=tolerance))


def assert_results_match(
    results_py: dict[int, npt.NDArray[np.floating]],
    results_cy: dict[int, npt.NDArray[np.floating]],
    tracker_name: str,
    perf_py: dict[str, float] | None = None,
    perf_cy: dict[str, float] | None = None,
) -> None:
    all_frames = sorted(set(results_py) | set(results_cy))
    empty = np.empty((0, 5), dtype=np.float64)
    n_differ = 0
    first_diff_frame: int | None = None

    for frame_idx in all_frames:
        a = results_py.get(frame_idx, empty)
        b = results_cy.get(frame_idx, empty)
        if not _compare_frames(a, b, TOLERANCE):
            n_differ += 1
            if first_diff_frame is None:
                first_diff_frame = frame_idx

    if perf_py and perf_cy:
        speedup = perf_py["total_time"] / perf_cy["total_time"] if perf_cy["total_time"] > 0 else 0
        print(
            f"\n{tracker_name}: {len(all_frames)} frames | "
            f"Python {perf_py['avg_ms']:.3f}ms/frame ({perf_py['fps']:.0f} fps) | "
            f"Cython {perf_cy['avg_ms']:.3f}ms/frame ({perf_cy['fps']:.0f} fps) | "
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

def test_sort_comparison():
    detection_results = load_detection_results(DETECTION_PATH)

    from references.sort.sort import Sort as SortPython, KalmanBoxTracker
    from pyxtrackers.sort.sort import Sort as SortCython

    def update(tracker, dets):
        return tracker.update(dets)

    KalmanBoxTracker.count = 0
    results_py, perf_py = run_tracker(
        SortPython(max_age=20, min_hits=1, iou_threshold=0.1),
        detection_results, update,
    )

    KalmanBoxTracker.count = 0
    results_cy, perf_cy = run_tracker(
        SortCython(max_age=20, min_hits=1, iou_threshold=0.1),
        detection_results, update,
    )

    assert_results_match(results_py, results_cy, "SORT", perf_py, perf_cy)


def test_ocsort_comparison():
    detection_results = load_detection_results(DETECTION_PATH)

    from references.ocsort.ocsort import OCSort as OCSortPython, KalmanBoxTracker
    from pyxtrackers.ocsort.ocsort import OCSort as OCSortCython

    img_info, img_size = (1080, 1920), (1080, 1920)

    def update_python(tracker, dets):
        return tracker.update(dets, img_info, img_size)

    def update_cython(tracker, dets):
        return tracker.update(dets)

    KalmanBoxTracker.count = 0
    results_py, perf_py = run_tracker(
        OCSortPython(det_thresh=0.3),
        detection_results, update_python,
    )

    KalmanBoxTracker.count = 0
    results_cy, perf_cy = run_tracker(
        OCSortCython(det_thresh=0.3),
        detection_results, update_cython,
    )

    assert_results_match(results_py, results_cy, "OC-SORT", perf_py, perf_cy)


def test_bytetrack_comparison():
    detection_results = load_detection_results(DETECTION_PATH)

    from references.bytetrack.byte_tracker import BYTETracker as BYTETrackerPython
    from references.bytetrack.basetrack import BaseTrack
    from pyxtrackers.bytetrack.bytetrack import BYTETracker as BYTETrackerCython

    img_info, img_size = (1080, 1920), (1080, 1920)

    def update_python(tracker, dets):
        tracked_objs = tracker.update(dets, img_info, img_size)
        if isinstance(tracked_objs, np.ndarray):
            return tracked_objs
        if len(tracked_objs) > 0:
            return np.array(
                [[t.tlbr[0], t.tlbr[1], t.tlbr[2], t.tlbr[3], t.track_id] for t in tracked_objs],
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

    BaseTrack._count = 0
    results_py, perf_py = run_tracker(
        BYTETrackerPython(Args()), detection_results, update_python,
    )

    BaseTrack._count = 0
    results_cy, perf_cy = run_tracker(
        BYTETrackerCython(track_thresh=0.5, match_thresh=0.8, track_buffer=30, mot20=False),
        detection_results, update_cython,
    )

    assert_results_match(results_py, results_cy, "ByteTrack", perf_py, perf_cy)
