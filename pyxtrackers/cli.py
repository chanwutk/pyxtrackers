"""
CLI interface for pyxtrackers.

Reads detections from stdin (one frame per line), runs a tracker, and writes
tracked objects to stdout (one frame per line). This enables any language to
use pyxtrackers via stdin/stdout pipes.

Input format:  x1,y1,x2,y2,score x1,y1,x2,y2,score ...
Output format: id,x1,y1,x2,y2 id,x1,y1,x2,y2 ...
Empty line = no detections (tracker still advances state, outputs empty line).
EOF = exit.
"""

import argparse
import sys

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyxtrackers",
        description="Multi-object tracking via stdin/stdout pipes.",
    )
    subparsers = parser.add_subparsers(dest="tracker", required=True)

    # --- SORT ---
    sort_parser = subparsers.add_parser("sort", help="SORT tracker")
    sort_parser.add_argument("--max-age", type=int, default=1)
    sort_parser.add_argument("--min-hits", type=int, default=3)
    sort_parser.add_argument("--iou-threshold", type=float, default=0.3)

    # --- ByteTrack ---
    bt_parser = subparsers.add_parser("bytetrack", help="ByteTrack tracker")
    bt_parser.add_argument("--track-thresh", type=float, default=0.5)
    bt_parser.add_argument("--match-thresh", type=float, default=0.8)
    bt_parser.add_argument("--track-buffer", type=int, default=30)
    bt_parser.add_argument("--mot20", action="store_true", default=False)
    bt_parser.add_argument("--frame-rate", type=int, default=30)
    bt_parser.add_argument("--img-info", type=float, nargs=2, default=None,
                           metavar=("H", "W"), help="Image info (height width)")
    bt_parser.add_argument("--img-size", type=float, nargs=2, default=None,
                           metavar=("H", "W"), help="Image size (height width)")

    # --- OC-SORT ---
    oc_parser = subparsers.add_parser("ocsort", help="OC-SORT tracker")
    oc_parser.add_argument("--det-thresh", type=float, default=0.3)
    oc_parser.add_argument("--max-age", type=int, default=30)
    oc_parser.add_argument("--min-hits", type=int, default=3)
    oc_parser.add_argument("--iou-threshold", type=float, default=0.3)
    oc_parser.add_argument("--delta-t", type=int, default=3)
    oc_parser.add_argument("--asso-func", type=str, default="iou")
    oc_parser.add_argument("--inertia", type=float, default=0.2)
    oc_parser.add_argument("--use-byte", action="store_true", default=False)
    oc_parser.add_argument("--img-info", type=float, nargs=2, default=None,
                           metavar=("H", "W"), help="Image info (height width)")
    oc_parser.add_argument("--img-size", type=float, nargs=2, default=None,
                           metavar=("H", "W"), help="Image size (height width)")

    return parser


def parse_detections(line: str) -> np.ndarray:
    """Parse a line of detections into an Nx5 numpy array.

    Each detection is ``x1,y1,x2,y2,score`` and detections are
    separated by spaces.  Returns an empty (0, 5) array for blank
    or whitespace-only lines.
    """
    line = line.strip()
    if not line:
        return np.empty((0, 5), dtype=np.float64)

    dets = []
    for token in line.split():
        parts = token.split(",")
        if len(parts) != 5:
            raise ValueError(
                f"Expected 5 comma-separated values per detection, got {len(parts)}: {token!r}"
            )
        dets.append([float(p) for p in parts])

    return np.array(dets, dtype=np.float64)


def format_tracks(tracks: np.ndarray) -> str:
    """Format tracked output as ``id,x1,y1,x2,y2 ...``."""
    if tracks.size == 0:
        return ""
    parts = []
    for row in tracks:
        x1, y1, x2, y2, tid = row
        parts.append(f"{int(tid)},{x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f}")
    return " ".join(parts)


def _create_tracker(args: argparse.Namespace):
    """Instantiate the appropriate tracker from parsed CLI args."""
    if args.tracker == "sort":
        from pyxtrackers.sort.sort import Sort
        return Sort(
            max_age=args.max_age,
            min_hits=args.min_hits,
            iou_threshold=args.iou_threshold,
        )

    if args.tracker == "bytetrack":
        from pyxtrackers.bytetrack.bytetrack import BYTETracker
        img_info = tuple(args.img_info) if args.img_info else None
        img_size = tuple(args.img_size) if args.img_size else None
        return BYTETracker(
            track_thresh=args.track_thresh,
            match_thresh=args.match_thresh,
            track_buffer=args.track_buffer,
            mot20=args.mot20,
            frame_rate=args.frame_rate,
            img_info=img_info,
            img_size=img_size,
        )

    if args.tracker == "ocsort":
        from pyxtrackers.ocsort.ocsort import OCSort
        img_info = tuple(args.img_info) if args.img_info else None
        img_size = tuple(args.img_size) if args.img_size else None
        return OCSort(
            det_thresh=args.det_thresh,
            max_age=args.max_age,
            min_hits=args.min_hits,
            iou_threshold=args.iou_threshold,
            delta_t=args.delta_t,
            asso_func=args.asso_func,
            inertia=args.inertia,
            use_byte=args.use_byte,
            img_info=img_info,
            img_size=img_size,
        )

    raise ValueError(f"Unknown tracker: {args.tracker}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    tracker = _create_tracker(args)

    while True:
        line = sys.stdin.readline()
        if not line:
            break

        dets = parse_detections(line)
        tracks = tracker.update(dets)
        print(format_tracks(tracks), flush=True)


if __name__ == "__main__":
    main()
