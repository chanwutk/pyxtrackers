"""Tests for pyxtrackers CLI."""

import subprocess
import sys

import numpy as np
import pytest

from pyxtrackers.cli import parse_detections, format_tracks, build_parser, _create_tracker


# ============================================================
# Unit tests: parse_detections
# ============================================================

class TestParseDetections:
    def test_normal(self):
        line = "100,200,300,400,0.9 150,250,350,450,0.8"
        result = parse_detections(line)
        assert result.shape == (2, 5)
        np.testing.assert_allclose(result[0], [100, 200, 300, 400, 0.9])
        np.testing.assert_allclose(result[1], [150, 250, 350, 450, 0.8])

    def test_single_detection(self):
        line = "10,20,30,40,0.5"
        result = parse_detections(line)
        assert result.shape == (1, 5)
        np.testing.assert_allclose(result[0], [10, 20, 30, 40, 0.5])

    def test_empty_line(self):
        result = parse_detections("")
        assert result.shape == (0, 5)

    def test_whitespace_only(self):
        result = parse_detections("   \n")
        assert result.shape == (0, 5)

    def test_malformed_too_few_values(self):
        with pytest.raises(ValueError, match="Expected 5"):
            parse_detections("100,200,300")

    def test_malformed_too_many_values(self):
        with pytest.raises(ValueError, match="Expected 5"):
            parse_detections("100,200,300,400,0.9,extra")

    def test_trailing_whitespace(self):
        line = "100,200,300,400,0.9  "
        result = parse_detections(line)
        assert result.shape == (1, 5)

    def test_float_precision(self):
        line = "100.123,200.456,300.789,400.012,0.99999"
        result = parse_detections(line)
        np.testing.assert_allclose(result[0], [100.123, 200.456, 300.789, 400.012, 0.99999])


# ============================================================
# Unit tests: format_tracks
# ============================================================

class TestFormatTracks:
    def test_normal(self):
        tracks = np.array([
            [100.0, 200.0, 300.0, 400.0, 1.0],
            [150.0, 250.0, 350.0, 450.0, 2.0],
        ])
        result = format_tracks(tracks)
        assert result == "1,100.0000,200.0000,300.0000,400.0000 2,150.0000,250.0000,350.0000,450.0000"

    def test_empty(self):
        tracks = np.empty((0, 5))
        result = format_tracks(tracks)
        assert result == ""

    def test_single_track(self):
        tracks = np.array([[10.5, 20.5, 30.5, 40.5, 3.0]])
        result = format_tracks(tracks)
        assert result == "3,10.5000,20.5000,30.5000,40.5000"

    def test_track_id_is_integer(self):
        tracks = np.array([[0, 0, 10, 10, 5.0]])
        result = format_tracks(tracks)
        assert result.startswith("5,")


# ============================================================
# Unit tests: build_parser / _create_tracker
# ============================================================

class TestBuildParser:
    def test_sort_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["sort"])
        assert args.tracker == "sort"
        assert args.max_age == 1
        assert args.min_hits == 3
        assert args.iou_threshold == 0.3

    def test_bytetrack_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["bytetrack"])
        assert args.tracker == "bytetrack"
        assert args.track_thresh == 0.5
        assert args.img_info is None
        assert args.img_size is None

    def test_bytetrack_with_img_info(self):
        parser = build_parser()
        args = parser.parse_args(["bytetrack", "--img-info", "1080", "1920", "--img-size", "1080", "1920"])
        assert args.img_info == [1080.0, 1920.0]
        assert args.img_size == [1080.0, 1920.0]

    def test_ocsort_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["ocsort"])
        assert args.tracker == "ocsort"
        assert args.det_thresh == 0.3
        assert args.use_byte is False

    def test_ocsort_use_byte(self):
        parser = build_parser()
        args = parser.parse_args(["ocsort", "--use-byte"])
        assert args.use_byte is True

    def test_no_tracker_fails(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


class TestCreateTracker:
    def test_create_sort(self):
        parser = build_parser()
        args = parser.parse_args(["sort", "--min-hits", "1"])
        tracker = _create_tracker(args)
        # Just verify it was created and can accept an update
        result = tracker.update(np.empty((0, 5), dtype=np.float64))
        assert isinstance(result, np.ndarray)

    def test_create_bytetrack(self):
        parser = build_parser()
        args = parser.parse_args(["bytetrack"])
        tracker = _create_tracker(args)
        result = tracker.update(np.empty((0, 5), dtype=np.float64))
        assert isinstance(result, np.ndarray)

    def test_create_ocsort(self):
        parser = build_parser()
        args = parser.parse_args(["ocsort"])
        tracker = _create_tracker(args)
        result = tracker.update(np.empty((0, 5), dtype=np.float64))
        assert isinstance(result, np.ndarray)


# ============================================================
# Integration tests: subprocess
# ============================================================

class TestCLIIntegration:
    """Integration tests that invoke the CLI as a subprocess."""

    def _run_cli(self, args: list[str], input_text: str, timeout: float = 30) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "pyxtrackers.cli"] + args,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def test_sort_help(self):
        result = self._run_cli(["sort", "--help"], "")
        assert result.returncode == 0
        assert "max-age" in result.stdout

    def test_bytetrack_help(self):
        result = self._run_cli(["bytetrack", "--help"], "")
        assert result.returncode == 0
        assert "track-thresh" in result.stdout

    def test_ocsort_help(self):
        result = self._run_cli(["ocsort", "--help"], "")
        assert result.returncode == 0
        assert "det-thresh" in result.stdout

    def test_sort_line_correspondence(self):
        """Output line count must match input line count."""
        input_lines = [
            "100,200,300,400,0.9 150,250,350,450,0.8",
            "105,205,305,405,0.9 155,255,355,455,0.8",
            "",  # empty frame
            "110,210,310,410,0.9",
        ]
        input_text = "\n".join(input_lines) + "\n"
        result = self._run_cli(["sort", "--min-hits", "1"], input_text)
        assert result.returncode == 0
        output_lines = result.stdout.strip("\n").split("\n")
        assert len(output_lines) == len(input_lines)

    def test_bytetrack_line_correspondence(self):
        input_lines = [
            "100,200,300,400,0.9",
            "105,205,305,405,0.9",
        ]
        input_text = "\n".join(input_lines) + "\n"
        result = self._run_cli(["bytetrack"], input_text)
        assert result.returncode == 0
        output_lines = result.stdout.strip("\n").split("\n")
        assert len(output_lines) == len(input_lines)

    def test_ocsort_line_correspondence(self):
        input_lines = [
            "100,200,300,400,0.9",
            "105,205,305,405,0.9",
        ]
        input_text = "\n".join(input_lines) + "\n"
        result = self._run_cli(["ocsort", "--min-hits", "1"], input_text)
        assert result.returncode == 0
        output_lines = result.stdout.strip("\n").split("\n")
        assert len(output_lines) == len(input_lines)

    def test_malformed_input_raises_error(self):
        input_text = "bad,data\n"
        result = self._run_cli(["sort"], input_text)
        assert result.returncode != 0
        assert "ValueError" in result.stderr

    def test_sort_round_trip(self):
        """Compare CLI output against direct Python tracker.update() output."""
        from pyxtrackers.sort.sort import Sort

        dets_per_frame = [
            np.array([[100, 200, 300, 400, 0.9], [150, 250, 350, 450, 0.8]], dtype=np.float64),
            np.array([[105, 205, 305, 405, 0.9], [155, 255, 355, 455, 0.8]], dtype=np.float64),
            np.array([[110, 210, 310, 410, 0.9]], dtype=np.float64),
        ]

        # Direct Python
        tracker = Sort(max_age=1, min_hits=1, iou_threshold=0.3)
        python_results = []
        for dets in dets_per_frame:
            python_results.append(tracker.update(dets))

        # CLI
        input_lines = []
        for dets in dets_per_frame:
            parts = []
            for row in dets:
                parts.append(",".join(str(v) for v in row))
            input_lines.append(" ".join(parts))
        input_text = "\n".join(input_lines) + "\n"

        result = self._run_cli(["sort", "--max-age", "1", "--min-hits", "1", "--iou-threshold", "0.3"], input_text)
        assert result.returncode == 0

        cli_lines = result.stdout.strip("\n").split("\n")
        assert len(cli_lines) == len(dets_per_frame)

        for i, (py_tracks, cli_line) in enumerate(zip(python_results, cli_lines)):
            if py_tracks.size == 0:
                assert cli_line.strip() == ""
                continue
            # Parse CLI output back to array
            cli_tracks = []
            for token in cli_line.strip().split():
                parts = token.split(",")
                tid = int(parts[0])
                x1, y1, x2, y2 = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                cli_tracks.append([x1, y1, x2, y2, tid])
            cli_arr = np.array(cli_tracks, dtype=np.float64)

            # Sort both by track ID for comparison
            py_sorted = py_tracks[py_tracks[:, 4].argsort()]
            cli_sorted = cli_arr[cli_arr[:, 4].argsort()]

            np.testing.assert_array_equal(py_sorted[:, 4], cli_sorted[:, 4],
                                          err_msg=f"Track IDs differ at frame {i}")
            np.testing.assert_allclose(py_sorted[:, :4], cli_sorted[:, :4], atol=0.01,
                                       err_msg=f"Bboxes differ at frame {i}")
