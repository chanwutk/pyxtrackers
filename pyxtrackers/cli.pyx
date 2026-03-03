"""
CLI interface for pyxtrackers.

Reads detections from stdin (one frame per line), runs a tracker, and writes
tracked objects to stdout (one frame per line). This enables any language to
use pyxtrackers via stdin/stdout pipes.

Input format:  x1,y1,x2,y2,score x1,y1,x2,y2,score ...
Output format: x1,y1,x2,y2,id x1,y1,x2,y2,id ...
Empty line = no detections (tracker still advances state, outputs empty line).
EOF = exit.

This file is a Cython-optimized implementation of references/cli.py.
Core flow/reference anchors:
  - Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/cli.py#L20-L60 (build_parser)
  - Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/cli.py#L63-L84 (parse_detections)
  - Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/cli.py#L86-L94 (format_tracks)
  - Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/cli.py#L97-L130 (_create_tracker)
  - Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/cli.py#L133-L155 (main loop)
"""

import argparse

import numpy as np
cimport numpy as cnp

# CPython C-API: create a Python bytes object from a C char* and explicit length,
# avoiding the overhead of Python-level bytes() construction.
from cpython.bytes cimport PyBytes_FromStringAndSize
# libc errno for detecting numeric overflow (ERANGE) from strtod.
from libc.errno cimport errno, ERANGE
# libc memory management and C-level numeric parsing (strtod).
from libc.stdlib cimport free, malloc, realloc, strtod
# libc stdio for C-level stdin/stdout I/O, bypassing Python's I/O stack.
from libc.stdio cimport FILE, feof, fgets, fflush, fwrite, snprintf, stdin, stdout
# libc string for strlen, used to measure raw C-string length.
from libc.string cimport strlen


# Typedef for numpy float64 typed memoryview access.
ctypedef cnp.float64_t float64_t

# 1 MiB read buffer for fgets. Each stdin line (one frame of detections)
# must fit within this limit; lines exceeding it trigger a ValueError.
cdef Py_ssize_t _READ_BUFFER_SIZE = 1024 * 1024


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with subcommands for each tracker.

    Each subcommand exposes the constructor parameters of the corresponding
    tracker class. Defaults here match the reference implementation defaults.
    Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/cli.py#L20-L60 (build_parser)
    """
    parser = argparse.ArgumentParser(
        prog="pyxtrackers",
        description="Multi-object tracking via stdin/stdout pipes.",
    )
    subparsers = parser.add_subparsers(dest="tracker", required=True)

    # --- SORT subcommand ---
    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/cli.py#L27-L32 (SORT CLI args/defaults)
    sort_parser = subparsers.add_parser("sort", help="SORT tracker")
    # Frames before a lost track is removed.
    sort_parser.add_argument("--max-age", type=int, default=1)
    # Minimum consecutive hits before a track is reported in output.
    sort_parser.add_argument("--min-hits", type=int, default=3)
    # IOU threshold used by matching.
    sort_parser.add_argument("--iou-threshold", type=float, default=0.3)

    # --- ByteTrack subcommand ---
    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/cli.py#L33-L44 (ByteTrack CLI args/defaults)
    bt_parser = subparsers.add_parser("bytetrack", help="ByteTrack tracker")
    # Detection confidence threshold.
    bt_parser.add_argument("--track-thresh", type=float, default=0.5)
    # IOU threshold used by the first-round association.
    bt_parser.add_argument("--match-thresh", type=float, default=0.8)
    # Buffer length for retaining lost tracks.
    bt_parser.add_argument("--track-buffer", type=int, default=30)
    # MOT20 mode toggle.
    bt_parser.add_argument("--mot20", action="store_true", default=False)
    # Frame rate setting used by tracker buffering logic.
    bt_parser.add_argument("--frame-rate", type=int, default=30)
    # NOTE: references/cli.py exposes --img-info/--img-size here
    # (references/cli.py#L40-L43), but pyxtrackers keeps them disabled.
    # Input format here assumes detections are already in pixel coordinates.
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

    # --- OC-SORT subcommand ---
    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/cli.py#L45-L58 (OC-SORT CLI args/defaults)
    oc_parser = subparsers.add_parser("ocsort", help="OC-SORT tracker")
    # Detection score threshold.
    oc_parser.add_argument("--det-thresh", type=float, default=0.3)
    # Maximum frames a track survives without a matching detection.
    oc_parser.add_argument("--max-age", type=int, default=30)
    # Minimum consecutive hits before a track appears in output.
    oc_parser.add_argument("--min-hits", type=int, default=3)
    # IOU threshold used in association steps.
    oc_parser.add_argument("--iou-threshold", type=float, default=0.3)
    # Number of past frames used by observation-centric recovery.
    oc_parser.add_argument("--delta-t", type=int, default=3)
    # Association function name.
    oc_parser.add_argument("--asso-func", type=str, default="iou")
    # Weight for velocity-based cost blending.
    oc_parser.add_argument("--inertia", type=float, default=0.2)
    # Enable second-round low-score association.
    oc_parser.add_argument("--use-byte", action="store_true", default=False)
    # NOTE: references/cli.py exposes --img-info/--img-size here
    # (references/cli.py#L55-L58), but pyxtrackers keeps them disabled.
    # Same rationale as ByteTrack above -- CLI input is pre-scaled pixels.
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
    """Check if a C char is ASCII whitespace.

    Used by the C-level detection parser (_parse_detections_bytes) to skip
    whitespace between detection tokens and detect end-of-content. This avoids
    calling into Python's str.isspace() from the inner parse loop.
    Covers all six standard ASCII whitespace characters: space, tab, newline,
    carriage return, vertical tab, and form feed.
    """
    return (
        ch == ' '
        or ch == '\t'
        or ch == '\n'
        or ch == '\r'
        or ch == '\v'
        or ch == '\f'
    )


cdef inline bytes _line_to_bytes(char* line_buf):
    """Convert a C char* (from fgets) to a Python bytes object.

    Uses CPython's PyBytes_FromStringAndSize to create a bytes object from the
    raw C buffer without going through Python's encode/decode machinery. The
    length is measured via strlen, so the result includes all characters up to
    (but not including) the null terminator -- including the trailing newline
    that fgets preserves.
    """
    cdef Py_ssize_t line_len = <Py_ssize_t>strlen(line_buf)
    return <bytes>PyBytes_FromStringAndSize(line_buf, line_len)


cdef inline cnp.ndarray[float64_t, ndim=2] _empty_dets():
    """Return an empty (0, 5) float64 detection array.

    All three trackers expect an Nx5 array where N can be 0 for frames with
    no detections. The shape (0, 5) matches the column layout
    [x1, y1, x2, y2, score].
    Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/cli.py#L71-L73 (blank/whitespace line -> empty (0, 5))
    """
    return np.empty((0, 5), dtype=np.float64)


cdef cnp.ndarray[float64_t, ndim=2] _parse_detections_bytes(bytes line_bytes):
    """Parse a raw bytes line into an Nx5 detection array using C-level strtod.

    Input format: "x1,y1,x2,y2,score x1,y1,x2,y2,score ..."
    where detections are whitespace-separated and fields within each detection
    are comma-separated. This is the hot path of CLI input parsing.

    Uses malloc/realloc for a growable C double buffer to avoid Python list
    append overhead. The buffer starts at capacity=8 detections and doubles
    when full. Values are parsed directly from the byte string via libc strtod,
    which is significantly faster than Python's float() or split().

    Fail-fast semantics: any malformed token raises ValueError immediately.
    Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/cli.py#L63-L84 (parse_detections behavior and validation)

    Returns an Nx5 numpy float64 array. If the line is empty or all whitespace,
    returns _empty_dets() (shape (0, 5)).
    """
    # Pointer arithmetic bounds: start and end delimit the input byte string.
    cdef const char* start = line_bytes
    cdef const char* end = start + len(line_bytes)
    cdef const char* p = start
    # parse_end is set by strtod to point past the last character consumed.
    cdef char* parse_end
    # Flat C array holding parsed doubles in row-major order (n_dets * 5).
    cdef double* values = NULL
    cdef Py_ssize_t n_dets = 0
    # Initial capacity for 8 detections (40 doubles); doubles on overflow.
    cdef Py_ssize_t capacity = 8
    cdef Py_ssize_t row_idx
    cdef Py_ssize_t col_idx
    # Typed memoryview for zero-overhead numpy array writes in the copy loop.
    cdef double[:, ::1] out_view
    cdef cnp.ndarray[float64_t, ndim=2] out
    cdef double parsed
    cdef double* resized

    # Skip leading whitespace (e.g., newline from previous fgets).
    while p < end and _is_ascii_whitespace(p[0]):
        p += 1

    # If the entire line is whitespace (or empty), return an empty array.
    # This handles the "empty line = no detections" contract.
    if p >= end:
        return _empty_dets()

    # Allocate initial buffer for detection values (capacity * 5 doubles).
    values = <double*>malloc(<size_t>(capacity * 5 * sizeof(double)))
    if values == NULL:
        raise MemoryError("Failed to allocate detection buffer")

    try:
        while True:
            # Grow buffer if we've filled it. Doubling amortizes to O(1) per detection.
            if n_dets == capacity:
                capacity *= 2
                resized = <double*>realloc(values, <size_t>(capacity * 5 * sizeof(double)))
                if resized == NULL:
                    raise MemoryError("Failed to grow detection buffer")
                values = resized

            # Parse exactly 5 comma-separated values for one detection token:
            # x1,y1,x2,y2,score -- matching the column layout expected by all
            # three tracker update() methods.
            for col_idx in range(5):
                # Reset errno before strtod to distinguish ERANGE from prior errors.
                errno = 0
                # strtod parses a double from p and advances parse_end past it.
                parsed = strtod(p, &parse_end)
                # If parse_end == p, strtod consumed nothing -- the input is
                # non-numeric or prematurely ended.
                if parse_end == p:
                    if p >= end or _is_ascii_whitespace(p[0]):
                        raise ValueError("Expected 5 comma-separated values per detection")
                    raise ValueError("Invalid numeric value in detection token")
                # ERANGE means overflow/underflow; we accept the clamped value
                # (inf or 0.0) and clear errno for subsequent calls.
                if errno == ERANGE:
                    errno = 0

                # Store parsed value into the flat buffer at [n_dets, col_idx].
                values[n_dets * 5 + col_idx] = parsed
                p = parse_end

                # Expect a comma between fields (but not after the 5th field).
                if col_idx < 4:
                    if p >= end or p[0] != ',':
                        raise ValueError("Expected 5 comma-separated values per detection")
                    p += 1  # Skip the comma.

            n_dets += 1

            # After a complete detection, check if we've reached end of input.
            if p >= end:
                break

            # Between detections, expect whitespace as the separator.
            if not _is_ascii_whitespace(p[0]):
                if p[0] == ',':
                    raise ValueError("Expected 5 comma-separated values per detection")
                raise ValueError("Unexpected characters after detection token")

            # Skip inter-detection whitespace.
            while p < end and _is_ascii_whitespace(p[0]):
                p += 1
            # Trailing whitespace at end of line is acceptable.
            if p >= end:
                break

        # Copy from the flat C buffer into a numpy array.
        # Using a typed memoryview (out_view) for the copy avoids Python-level
        # indexing overhead on each element.
        out = np.empty((n_dets, 5), dtype=np.float64)
        out_view = out
        for row_idx in range(n_dets):
            for col_idx in range(5):
                out_view[row_idx, col_idx] = values[row_idx * 5 + col_idx]
        return out
    finally:
        # Always free the C buffer, even if an exception was raised.
        if values != NULL:
            free(values)


def parse_detections(str line):
    """Parse a line of detections into an Nx5 numpy array.

    Python-callable wrapper around the C-level _parse_detections_bytes.
    Encodes the input string to ASCII bytes first, since strtod operates on
    raw byte data. This is the public API; the CLI main loop calls
    _parse_detections_bytes directly for performance (skips the encode step
    because fgets already produces raw bytes).
    Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/cli.py#L63-L84 (parse_detections)
    """
    return _parse_detections_bytes(line.encode("ascii"))


def format_tracks(cnp.ndarray[float64_t, ndim=2] tracks):
    """Format tracked output as ``x1,y1,x2,y2,id ...`` (Python-level).

    Output schema: space-separated tokens, each "x1,y1,x2,y2,id" where
    coordinates are 6-decimal floats and id is an integer. This matches the
    CLI output contract where track_id is the last field (matching the Nx5
    numpy column order [x1, y1, x2, y2, track_id] returned by all three
    tracker update() methods).

    This is the pure-Python formatting path, used by tests and Python callers.
    The CLI main loop uses _write_tracks_stdout instead for C-level I/O.
    Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/cli.py#L86-L94 (format_tracks)
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t n = tracks.shape[0]
    cdef list parts
    # Return empty string for zero tracks, preserving 1:1 line correspondence.
    if tracks.size == 0:
        return ""

    parts = []
    for i in range(n):
        # Format: x1,y1,x2,y2,id -- coordinates as 6-decimal floats, id as int.
        parts.append(
            f"{tracks[i, 0]:.6f},{tracks[i, 1]:.6f},{tracks[i, 2]:.6f},{tracks[i, 3]:.6f},{int(tracks[i, 4])}"
        )
    return " ".join(parts)


cdef void _write_tracks_stdout(cnp.ndarray[float64_t, ndim=2] tracks) except *:
    """Write tracked output directly to C stdout using snprintf + fwrite.

    This is the C-level fast path for CLI output, bypassing Python's I/O stack.
    It formats each track as "x1,y1,x2,y2,id" (same schema as format_tracks),
    space-separates them, appends a newline, and writes the entire frame in one
    fwrite call followed by fflush for immediate pipe delivery.

    Uses malloc'd buffer with 128 bytes per row (enough for 4 floats with
    %.6f precision + 1 integer + commas + space separator).

    Preserves the 1:1 input/output line correspondence contract: even zero-track
    frames emit a newline.
    Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/cli.py#L86-L94 (same formatting/output contract)
    """
    cdef Py_ssize_t n = tracks.shape[0]
    cdef Py_ssize_t i
    cdef int written
    # 128 bytes per row is generous for "%.6f,%.6f,%.6f,%.6f,%d" format.
    cdef Py_ssize_t row_capacity
    cdef Py_ssize_t max_bytes
    cdef char* out_buf
    cdef char* out_ptr
    # Typed memoryview for direct C-level array access without Python indexing.
    cdef double[:, ::1] view

    # Zero tracks: emit just a newline to maintain 1:1 line correspondence.
    if tracks.size == 0:
        fwrite(b"\n", 1, 1, stdout)
        fflush(stdout)
        return

    # Allocate output buffer: 128 bytes per track + 1 byte for space separator
    # per track, + 1 byte for trailing newline.
    row_capacity = 128
    max_bytes = (row_capacity + 1) * n + 1
    out_buf = <char*>malloc(<size_t>max_bytes)
    if out_buf == NULL:
        raise MemoryError("Failed to allocate output buffer")

    try:
        out_ptr = out_buf
        view = tracks
        for i in range(n):
            # snprintf formats one track: "x1,y1,x2,y2,id" with %.6f precision
            # for coordinates and %d for the integer track ID.
            written = snprintf(
                out_ptr,
                <size_t>(row_capacity),
                b"%.6f,%.6f,%.6f,%.6f,%d",
                view[i, 0],
                view[i, 1],
                view[i, 2],
                view[i, 3],
                <int>view[i, 4],
            )
            # snprintf returns negative on encoding error.
            if written < 0:
                raise RuntimeError("Failed to format output row")
            # snprintf returns >= size when output was truncated.
            if written >= row_capacity:
                raise RuntimeError("Output row exceeded formatter capacity")
            out_ptr += written
            # Space-separate tracks (no trailing space before newline).
            if i != n - 1:
                out_ptr[0] = ' '
                out_ptr += 1

        # Terminate the line with a newline.
        out_ptr[0] = '\n'
        out_ptr += 1
        # Single fwrite for the entire frame, then flush for pipe consumers.
        fwrite(out_buf, 1, <size_t>(out_ptr - out_buf), stdout)
        fflush(stdout)
    finally:
        # Always free the output buffer, even on formatting errors.
        free(out_buf)


def _create_tracker(args: argparse.Namespace):
    """Instantiate the appropriate tracker from parsed CLI args.

    Lazy-imports each tracker to avoid loading all three Cython extensions
    when only one is needed. Parameters are forwarded from CLI args to
    tracker constructors.
    Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/cli.py#L97-L130 (_create_tracker)
    """
    if args.tracker == "sort":
        from pyxtrackers.sort.sort import Sort

        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/cli.py#L99-L105 (SORT branch in _create_tracker)
        return Sort(
            max_age=args.max_age,
            min_hits=args.min_hits,
            iou_threshold=args.iou_threshold,
        )

    if args.tracker == "bytetrack":
        from pyxtrackers.bytetrack.bytetrack import BYTETracker

        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/cli.py#L107-L115 (ByteTrack branch in _create_tracker)
        return BYTETracker(
            track_thresh=args.track_thresh,
            match_thresh=args.match_thresh,
            track_buffer=args.track_buffer,
            mot20=args.mot20,
            frame_rate=args.frame_rate,
        )

    if args.tracker == "ocsort":
        from pyxtrackers.ocsort.ocsort import OCSort

        # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/cli.py#L117-L128 (OC-SORT branch in _create_tracker)
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
    """Main CLI entry point: read stdin, track, write stdout.

    Processing loop:
      1. Read one line from stdin via C fgets (one frame of detections).
      2. Parse the line into an Nx5 numpy array via _parse_detections_bytes.
      3. Call tracker.update(dets) to advance tracker state and get active tracks.
         - Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/cli.py#L153-L154 (tracks = tracker.update(...))
      4. Write the Nx5 tracks array to stdout via _write_tracks_stdout.
      5. Repeat until EOF.

    Each tracker's update() must be called once per frame even with empty
    detections -- this advances internal state (frame counters, Kalman predict
    steps, track age management). The empty-line contract ensures this.
    Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/cli.py#L133-L155 (main)
    """
    # Use C FILE* for stdin to pair with fgets (bypasses Python's sys.stdin).
    cdef FILE* in_stream = stdin
    # malloc'd read buffer for fgets; freed in the finally block.
    cdef char* read_buf = NULL
    cdef bytes line_bytes
    # Typed arrays for the inner loop: detections in, tracks out.
    cdef cnp.ndarray[float64_t, ndim=2] dets
    cdef cnp.ndarray[float64_t, ndim=2] tracks

    # Parse CLI arguments and instantiate the selected tracker.
    # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/cli.py#L134-L136
    parser = build_parser()
    args = parser.parse_args(argv)
    tracker = _create_tracker(args)

    # NOTE: img_info/img_size scaling is commented out. The CLI assumes
    # detections are pre-scaled to pixel coordinates. references/cli.py wires
    # scaling via args.img_info/img_size
    # (references/cli.py#L138-L143, references/cli.py#L151-L152).
    # Our Cython reimplementation accepts pre-scaled input.
    # img_info = None
    # img_size = None
    # if hasattr(args, "img_info") and args.img_info and args.img_size:
    #     from pyxtrackers.utils.scale import scale as scale_dets
    #
    #     img_info = tuple(args.img_info)
    #     img_size = tuple(args.img_size)

    # Allocate the stdin read buffer (1 MiB).
    read_buf = <char*>malloc(<size_t>_READ_BUFFER_SIZE)
    if read_buf == NULL:
        raise MemoryError("Failed to allocate stdin read buffer")

    try:
        while True:
            # Read one line from stdin using C fgets. fgets reads up to
            # _READ_BUFFER_SIZE-1 chars and null-terminates. Returns NULL on
            # EOF or error, which breaks the loop.
            if fgets(read_buf, <int>_READ_BUFFER_SIZE, in_stream) == NULL:
                break

            # Overflow check: if fgets filled the entire buffer without finding
            # a newline and we haven't hit EOF, the line is too long. This is
            # a fail-fast guard -- better to error than silently truncate.
            if strlen(read_buf) == _READ_BUFFER_SIZE - 1 and read_buf[_READ_BUFFER_SIZE - 2] != '\n' and feof(in_stream) == 0:
                raise ValueError("Input line exceeds CLI read buffer")

            # Convert the raw C buffer to Python bytes (via PyBytes_FromStringAndSize),
            # then parse into an Nx5 detection array using the C-level parser.
            line_bytes = _line_to_bytes(read_buf)
            dets = _parse_detections_bytes(line_bytes)
            # NOTE: optional rescaling hook (disabled; see comment above).
            # if img_info is not None:
            #     dets = scale_dets(dets, img_info, img_size)

            # Call the tracker's update method. Each tracker expects an Nx5
            # float64 array with columns [x1, y1, x2, y2, score] and returns
            # an Mx5 array with columns [x1, y1, x2, y2, track_id].
            # Ref: https://github.com/chanwutk/pyxtrackers/blob/main/references/cli.py#L153-L154 (tracks = tracker.update(...), output)
            tracks = tracker.update(dets)
            # Write the tracks to stdout using C-level I/O for performance.
            _write_tracks_stdout(tracks)
    finally:
        # Always free the read buffer, even if an exception interrupted the loop.
        free(read_buf)
