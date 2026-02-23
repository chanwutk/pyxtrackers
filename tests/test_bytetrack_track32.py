#!/usr/bin/env python3
"""
Debug test to track what happens to track ID 32 specifically.
"""

import json
import os
import numpy as np

from references.bytetrack.byte_tracker import BYTETracker as BYTETrackerPython
from pyxtrackers.bytetrack.bytetrack import BYTETracker as BYTETrackerCython
from references.bytetrack.basetrack import BaseTrack


def test_track_32():
    """Track what happens to track ID 32."""

    # Path to detection results file
    detection_path = os.path.join(
        os.path.dirname(__file__), 'data', 'detection.jsonl'
    )

    # Load frames 660-670
    detection_results = []
    with open(detection_path, 'r') as f:
        for i, line in enumerate(f):
            if i < 650:
                continue
            if i >= 670:
                break
            if line.strip():
                detection_results.append(json.loads(line))

    # Create args
    class Args:
        track_thresh = 0.5
        track_buffer = 30
        match_thresh = 0.8
        mot20 = False

    args = Args()

    img_info = (1080, 1920)
    img_size = (1080, 1920)

    # Initialize Python tracker and run to frame 650
    BaseTrack._count = 0
    tracker_python = BYTETrackerPython(args)

    # Run Python tracker to frame 649 (to set up state)
    with open(detection_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 650:
                break
            if line.strip():
                frame_result = json.loads(line)
                detections = frame_result.get('detections', [])
                if len(detections) > 0:
                    dets = np.array(detections, dtype=np.float64)
                    if dets.shape[1] < 5:
                        scores = np.ones((dets.shape[0], 1), dtype=np.float64)
                        dets = np.concatenate([dets, scores], axis=1)
                    dets = dets[:, :5]
                else:
                    dets = np.empty((0, 5), dtype=np.float64)
                tracker_python.update(dets, img_info, img_size)

    # Run Cython tracker to frame 649
    tracker_cython = BYTETrackerCython(
        track_thresh=0.5, match_thresh=0.8, track_buffer=30,
        mot20=False, img_info=img_info, img_size=img_size,
    )

    with open(detection_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 650:
                break
            if line.strip():
                frame_result = json.loads(line)
                detections = frame_result.get('detections', [])
                if len(detections) > 0:
                    dets = np.array(detections, dtype=np.float64)
                    if dets.shape[1] < 5:
                        scores = np.ones((dets.shape[0], 1), dtype=np.float64)
                        dets = np.concatenate([dets, scores], axis=1)
                    dets = dets[:, :5]
                else:
                    dets = np.empty((0, 5), dtype=np.float64)
                tracker_cython.update(dets)

    # Now process frames 650-669 with detailed tracking
    for frame_result in detection_results:
        frame_idx = frame_result['frame_idx']
        detections = frame_result.get('detections', [])

        if len(detections) > 0:
            dets = np.array(detections, dtype=np.float64)
            if dets.shape[1] < 5:
                scores = np.ones((dets.shape[0], 1), dtype=np.float64)
                dets = np.concatenate([dets, scores], axis=1)
            dets = dets[:, :5]
        else:
            dets = np.empty((0, 5), dtype=np.float64)

        # Check if track 32 exists before update
        python_has_32_before = any(t.track_id == 32 for t in tracker_python.tracked_stracks + tracker_python.lost_stracks)
        cython_has_32_before = any(t.track_id == 32 for t in tracker_cython.tracked_stracks + tracker_cython.lost_stracks)

        # Run both trackers
        python_result = tracker_python.update(dets, img_info, img_size)
        cython_result = tracker_cython.update(dets)

        # Check track 32 status after update
        python_32_tracked = any(t.track_id == 32 for t in tracker_python.tracked_stracks)
        python_32_lost = any(t.track_id == 32 for t in tracker_python.lost_stracks)
        cython_32_tracked = any(t.track_id == 32 for t in tracker_cython.tracked_stracks)
        cython_32_lost = any(t.track_id == 32 for t in tracker_cython.lost_stracks)

        # Print info if track 32 exists in either tracker
        if python_has_32_before or cython_has_32_before or python_32_tracked or python_32_lost or cython_32_tracked or cython_32_lost:
            print(f"\n=== Frame {frame_idx} ===")
            print(f"Detections: {len(dets)}")
            print(f"\nPython - Track 32:")
            if python_32_tracked:
                track_32 = [t for t in tracker_python.tracked_stracks if t.track_id == 32][0]
                print(f"  Status: TRACKED")
                print(f"  Score: {track_32.score:.3f}")
                print(f"  State: {track_32.state}")
            elif python_32_lost:
                track_32 = [t for t in tracker_python.lost_stracks if t.track_id == 32][0]
                print(f"  Status: LOST")
                print(f"  State: {track_32.state}")
            else:
                print(f"  Status: NOT FOUND")

            print(f"\nCython - Track 32:")
            if cython_32_tracked:
                track_32 = [t for t in tracker_cython.tracked_stracks if t.track_id == 32][0]
                print(f"  Status: TRACKED")
                print(f"  Score: {track_32.score:.3f}")
                print(f"  State: {track_32.state}")
            elif cython_32_lost:
                track_32 = [t for t in tracker_cython.lost_stracks if t.track_id == 32][0]
                print(f"  Status: LOST")
                print(f"  State: {track_32.state}")
            else:
                print(f"  Status: NOT FOUND")


if __name__ == '__main__':
    test_track_32()
