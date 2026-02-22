# PyXTrackers

High-performance Cython implementations of state-of-the-art multi-object tracking algorithms.

PyXTrackers provides drop-in replacements for three widely used MOT trackers, reimplemented in Cython for significant speedups while maintaining numerical equivalence with the original Python implementations.

## Supported Trackers

| Tracker | Description | Paper | GitHub |
|---------|-------------|-------|--------|
| **SORT** | Simple Online and Realtime Tracking | [Bewley et al., 2016](https://arxiv.org/abs/1602.00763) | https://github.com/abewley/sort |
| **ByteTrack** | Multi-Object Tracking by Associating Every Detection Box | [Zhang et al., 2022](https://arxiv.org/abs/2110.06864) | https://github.com/FoundationVision/ByteTrack |
| **OC-SORT** | Observation-Centric SORT | [Cao et al., 2023](https://arxiv.org/abs/2203.14360) | https://github.com/noahcao/OC_SORT |

## Installation

### From source (with uv, recommended)

```bash
git clone https://github.com/chanwutk/pyxtrackers.git
cd pyxtrackers
uv sync
uv run python setup.py build_ext
```

### From source (with pip)

```bash
git clone https://github.com/chanwutk/pyxtrackers.git
cd pyxtrackers
pip install -e .
```

### Requirements

- Python >= 3.10
- NumPy >= 2.0
- A C/C++ compiler (gcc, clang, or MSVC)
- Cython >= 3.0 (build-time only)

## Quick Start

```python
import numpy as np
from pyxtrackers import Sort, BYTETracker, OCSort

# --- SORT ---
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# Detections: [[x1, y1, x2, y2, score], ...]
detections = np.array([
    [100, 100, 200, 200, 0.9],
    [300, 300, 400, 400, 0.8],
], dtype=np.float64)

# Returns: [[x1, y1, x2, y2, track_id], ...]
tracked = tracker.update(detections)

# --- ByteTrack ---
class Args:
    track_thresh = 0.5
    track_buffer = 30
    match_thresh = 0.8
    mot20 = False

tracker = BYTETracker(Args())
img_info = (1080, 1920)  # (height, width)
img_size = (1080, 1920)

online_targets = tracker.update(detections, img_info, img_size)
# Each target has: .tlwh, .track_id, .score

# --- OC-SORT ---
tracker = OCSort(img_size=(1080, 1920), det_thresh=0.3)

tracked = tracker.update(detections)
# Returns: [[x1, y1, x2, y2, track_id], ...]
```

## How It Works

Each tracker is reimplemented in Cython using:

- **C structs** instead of Python classes for track state (zero Python object overhead)
- **Flat arrays** for matrices (e.g., `double[64]` for 8x8 covariance) for cache-friendly layout
- **`nogil` sections** for GIL-free computation in hot paths
- **`cdef` functions** for C-only internal calls with no Python dispatch overhead
- **Vendored LAPJV** C++ solver for linear assignment (no external dependency)

Kalman filter predict/update cycles, IOU computation, and the Hungarian algorithm all run at C speed.

## Project Structure

```
pyxtrackers/           # Installable Cython package
  sort/                # SORT tracker (7D Kalman, constant velocity)
  bytetrack/           # ByteTrack tracker (8D Kalman, two-stage association)
  ocsort/              # OC-SORT tracker (7D Kalman, freeze/unfreeze)
references/            # Pure Python reference implementations (for testing)
tests/                 # Comparison tests verifying numerical equivalence
vendor/lapjv/          # Vendored C++ linear assignment solver
```

## Development

```bash
# Setup
uv sync
uv run python setup.py build_ext

# Run tests
uv run pytest tests/ -v

# Rebuild after changing .pyx files
uv run python setup.py build_ext

# Clean build artifacts
uv run python setup.py clean
```

## Testing

Tests run both the Cython and Python reference implementations on identical detection sequences and verify numerical equivalence within 1e-6 pixel tolerance.

```bash
uv run pytest
```

## Roadmap

- [ ] Clean up the tracker interface
- [ ] Add scalability tests (runtime when scaling the number of detections/frames, runtime per frame for different number of detections)
- [ ] Complete migration to uv for project management
- [ ] Add GitHub Actions CI to test across environments (pyenv, uv, conda, poetry)
- [ ] Add CLI support

## License

MIT
