"""
CLI interface for pyxtrackers.

Reads detections from stdin (one frame per line), runs a tracker, and writes
tracked objects to stdout (one frame per line). This enables any language to
use pyxtrackers via stdin/stdout pipes.

Input format:  x1,y1,x2,y2,score x1,y1,x2,y2,score ...
Output format: x1,y1,x2,y2,id x1,y1,x2,y2,id ...
Empty line = no detections (tracker still advances state, outputs empty line).
EOF = exit.
"""

import argparse

import numpy as np
cimport numpy as cnp

from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.errno cimport errno, ERANGE
from libc.stdlib cimport free, malloc, realloc, strtod
from libc.stdio cimport FILE, feof, fgets, fflush, fwrite, snprintf, stdin, stdout
from libc.string cimport strlen


ctypedef cnp.float64_t float64_t

cdef Py_ssize_t _READ_BUFFER_SIZE = 1024 * 1024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyxtrackers",
        description="Multi-object tracking via stdin/stdout pipes.",
    )
    subparsers = parser.add_subparsers(dest="tracker", required=True)

    sort_parser = subparsers.add_parser("sort", help="SORT tracker")
    sort_parser.add_argument("--max-age", type=int, default=1)
    sort_parser.add_argument("--min-hits", type=int, default=3)
    sort_parser.add_argument("--iou-threshold", type=float, default=0.3)

    bt_parser = subparsers.add_parser("bytetrack", help="ByteTrack tracker")
    bt_parser.add_argument("--track-thresh", type=float, default=0.5)
    bt_parser.add_argument("--match-thresh", type=float, default=0.8)
    bt_parser.add_argument("--track-buffer", type=int, default=30)
    bt_parser.add_argument("--mot20", action="store_true", default=False)
    bt_parser.add_argument("--frame-rate", type=int, default=30)
    # bt_parser.add_argument(
    #     "--img-info",
    #     type=float,
    #     nargs=2,
    #     default=None,
    #     metavar=("H", "W"),
    #     help="Image info (height width)",
    # )
    # bt_parser.add_argument(
    #     "--img-size",
    #     type=float,
    #     nargs=2,
    #     default=None,
    #     metavar=("H", "W"),
    #     help="Image size (height width)",
    # )

    oc_parser = subparsers.add_parser("ocsort", help="OC-SORT tracker")
    oc_parser.add_argument("--det-thresh", type=float, default=0.3)
    oc_parser.add_argument("--max-age", type=int, default=30)
    oc_parser.add_argument("--min-hits", type=int, default=3)
    oc_parser.add_argument("--iou-threshold", type=float, default=0.3)
    oc_parser.add_argument("--delta-t", type=int, default=3)
    oc_parser.add_argument("--asso-func", type=str, default="iou")
    oc_parser.add_argument("--inertia", type=float, default=0.2)
    oc_parser.add_argument("--use-byte", action="store_true", default=False)
    # oc_parser.add_argument(
    #     "--img-info",
    #     type=float,
    #     nargs=2,
    #     default=None,
    #     metavar=("H", "W"),
    #     help="Image info (height width)",
    # )
    # oc_parser.add_argument(
    #     "--img-size",
    #     type=float,
    #     nargs=2,
    #     default=None,
    #     metavar=("H", "W"),
    #     help="Image size (height width)",
    # )

    return parser


cdef inline bint _is_ascii_whitespace(char ch):
    return (
        ch == ' '
        or ch == '\t'
        or ch == '\n'
        or ch == '\r'
        or ch == '\v'
        or ch == '\f'
    )


cdef inline bytes _line_to_bytes(char* line_buf):
    cdef Py_ssize_t line_len = <Py_ssize_t>strlen(line_buf)
    return <bytes>PyBytes_FromStringAndSize(line_buf, line_len)


cdef inline cnp.ndarray[float64_t, ndim=2] _empty_dets():
    return np.empty((0, 5), dtype=np.float64)


cdef cnp.ndarray[float64_t, ndim=2] _parse_detections_bytes(bytes line_bytes):
    cdef const char* start = line_bytes
    cdef const char* end = start + len(line_bytes)
    cdef const char* p = start
    cdef char* parse_end
    cdef double* values = NULL
    cdef Py_ssize_t n_dets = 0
    cdef Py_ssize_t capacity = 8
    cdef Py_ssize_t row_idx
    cdef Py_ssize_t col_idx
    cdef double[:, ::1] out_view
    cdef cnp.ndarray[float64_t, ndim=2] out
    cdef double parsed
    cdef double* resized

    while p < end and _is_ascii_whitespace(p[0]):
        p += 1

    if p >= end:
        return _empty_dets()

    values = <double*>malloc(<size_t>(capacity * 5 * sizeof(double)))
    if values == NULL:
        raise MemoryError("Failed to allocate detection buffer")

    try:
        while True:
            if n_dets == capacity:
                capacity *= 2
                resized = <double*>realloc(values, <size_t>(capacity * 5 * sizeof(double)))
                if resized == NULL:
                    raise MemoryError("Failed to grow detection buffer")
                values = resized

            for col_idx in range(5):
                errno = 0
                parsed = strtod(p, &parse_end)
                if parse_end == p:
                    if p >= end or _is_ascii_whitespace(p[0]):
                        raise ValueError("Expected 5 comma-separated values per detection")
                    raise ValueError("Invalid numeric value in detection token")
                if errno == ERANGE:
                    errno = 0

                values[n_dets * 5 + col_idx] = parsed
                p = parse_end

                if col_idx < 4:
                    if p >= end or p[0] != ',':
                        raise ValueError("Expected 5 comma-separated values per detection")
                    p += 1

            n_dets += 1

            if p >= end:
                break

            if not _is_ascii_whitespace(p[0]):
                if p[0] == ',':
                    raise ValueError("Expected 5 comma-separated values per detection")
                raise ValueError("Unexpected characters after detection token")

            while p < end and _is_ascii_whitespace(p[0]):
                p += 1
            if p >= end:
                break

        out = np.empty((n_dets, 5), dtype=np.float64)
        out_view = out
        for row_idx in range(n_dets):
            for col_idx in range(5):
                out_view[row_idx, col_idx] = values[row_idx * 5 + col_idx]
        return out
    finally:
        if values != NULL:
            free(values)


def parse_detections(str line):
    """Parse a line of detections into an Nx5 numpy array."""
    return _parse_detections_bytes(line.encode("ascii"))


def format_tracks(cnp.ndarray[float64_t, ndim=2] tracks):
    """Format tracked output as ``x1,y1,x2,y2,id ...``."""
    cdef Py_ssize_t i
    cdef Py_ssize_t n = tracks.shape[0]
    cdef list parts
    if tracks.size == 0:
        return ""

    parts = []
    for i in range(n):
        parts.append(
            f"{tracks[i, 0]:.4f},{tracks[i, 1]:.4f},{tracks[i, 2]:.4f},{tracks[i, 3]:.4f},{int(tracks[i, 4])}"
        )
    return " ".join(parts)


cdef void _write_tracks_stdout(cnp.ndarray[float64_t, ndim=2] tracks) except *:
    cdef Py_ssize_t n = tracks.shape[0]
    cdef Py_ssize_t i
    cdef int written
    cdef Py_ssize_t row_capacity
    cdef Py_ssize_t max_bytes
    cdef char* out_buf
    cdef char* out_ptr
    cdef double[:, ::1] view

    if tracks.size == 0:
        fwrite(b"\n", 1, 1, stdout)
        fflush(stdout)
        return

    row_capacity = 96
    max_bytes = (row_capacity + 1) * n + 1
    out_buf = <char*>malloc(<size_t>max_bytes)
    if out_buf == NULL:
        raise MemoryError("Failed to allocate output buffer")

    try:
        out_ptr = out_buf
        view = tracks
        for i in range(n):
            written = snprintf(
                out_ptr,
                <size_t>(row_capacity),
                b"%.4f,%.4f,%.4f,%.4f,%d",
                view[i, 0],
                view[i, 1],
                view[i, 2],
                view[i, 3],
                <int>view[i, 4],
            )
            if written < 0:
                raise RuntimeError("Failed to format output row")
            if written >= row_capacity:
                raise RuntimeError("Output row exceeded formatter capacity")
            out_ptr += written
            if i != n - 1:
                out_ptr[0] = ' '
                out_ptr += 1

        out_ptr[0] = '\n'
        out_ptr += 1
        fwrite(out_buf, 1, <size_t>(out_ptr - out_buf), stdout)
        fflush(stdout)
    finally:
        free(out_buf)


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

        return BYTETracker(
            track_thresh=args.track_thresh,
            match_thresh=args.match_thresh,
            track_buffer=args.track_buffer,
            mot20=args.mot20,
            frame_rate=args.frame_rate,
        )

    if args.tracker == "ocsort":
        from pyxtrackers.ocsort.ocsort import OCSort

        return OCSort(
            det_thresh=args.det_thresh,
            max_age=args.max_age,
            min_hits=args.min_hits,
            iou_threshold=args.iou_threshold,
            delta_t=args.delta_t,
            asso_func=args.asso_func,
            inertia=args.inertia,
            use_byte=args.use_byte,
        )

    raise ValueError(f"Unknown tracker: {args.tracker}")


def main(argv: list[str] | None = None) -> None:
    cdef FILE* in_stream = stdin
    cdef char* read_buf = NULL
    cdef bytes line_bytes
    cdef cnp.ndarray[float64_t, ndim=2] dets
    cdef cnp.ndarray[float64_t, ndim=2] tracks

    parser = build_parser()
    args = parser.parse_args(argv)
    tracker = _create_tracker(args)

    # img_info = None
    # img_size = None
    # if hasattr(args, "img_info") and args.img_info and args.img_size:
    #     from pyxtrackers.utils.scale import scale as scale_dets
    #
    #     img_info = tuple(args.img_info)
    #     img_size = tuple(args.img_size)

    read_buf = <char*>malloc(<size_t>_READ_BUFFER_SIZE)
    if read_buf == NULL:
        raise MemoryError("Failed to allocate stdin read buffer")

    try:
        while True:
            if fgets(read_buf, <int>_READ_BUFFER_SIZE, in_stream) == NULL:
                break

            if strlen(read_buf) == _READ_BUFFER_SIZE - 1 and read_buf[_READ_BUFFER_SIZE - 2] != '\n' and feof(in_stream) == 0:
                raise ValueError("Input line exceeds CLI read buffer")

            line_bytes = _line_to_bytes(read_buf)
            dets = _parse_detections_bytes(line_bytes)
            # if img_info is not None:
            #     dets = scale_dets(dets, img_info, img_size)

            tracks = tracker.update(dets)
            _write_tracks_stdout(tracks)
    finally:
        free(read_buf)
