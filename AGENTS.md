Review this plan thoroughly before making any code changes. For every issue or recommendation, explain the concrete tradeoffs, give me an opinionated recommendation, and ask for my input before assuming a direction.
My engineering preferences (use these to guide your recommendations):
* DRY is important-flag repetition aggressively.
* Well-tested code is non-negotiable, I'd rather have too many tests than too few.
* I want code that's "engineered enough"--not under-engineered (fragile, hacky) and not over-engineered (premature abstraction, unnecessary complexity).
* l err on the side of handling more edge cases, not fewer; thoughtfulness > speed.
* Bias toward explicit over cleverness.

1. Architecture review
Evaluate:
  * Overall system design and component boundaries.
  * Dependency graph and coupling concerns.
  * Data flow patterns and potential bottlenecks.
  * Scaling characteristics and single points of failure.
  * Security architecture (auth, data access, API boundaries)
2. Code quality review
Evaluate:
  * Code organization and module structure.
  * DRY violations–be aggressive here.
  * Error handling patterns and missing edge cases (call these out explicitly).
  * Technical debt hotspots.
  * Areas that are over-engineered or under-engineered relative to my preferences.
3. Test review
Evaluate:
  * Test coverage gaps (unit, integration, e2e). test quay and assertion strength
  * Missing edge case coverage–be thorough.
  * Untested failure modes and error paths.
4. Performance review
Evaluate:
  * N+1 queries and database access patterns.
  * Memory-usage concerns.
  * Caching opportunities.
  * Slow or high-complexity code paths.

For each issue you find
For every specific issue (bug, smell, design concern, or risk):
* Describe the problem concretely, with file and line references.
* Presence-s options, including do nothing where that's reasonable.
* For each option, specify: implementation effort, risk, impact on other code, and maintenance burden.
* Give me your recommended option and why, mapped to my preferences above.
* Then explicitly ask whether i agree or want to choose a different direction before proceeding
Workflow and interaction
* Do not assume my priorities on timeline or scale.
* After each section, pause and ask for my feedback before moving on.

BEFORE YOU START:
Ask it I want one of two options:
1/ BIG CHANGE: Work through this interactively, one section at a time (Architecture → Code Quality → Tests → Performance) with at most 4 top issues in each section.
SMALL CHANGE: work through interactively oNe question per review section

FOR EACH STAGE OF REVIEW: output the explanation and pros and cons of each stage's questions AND your opinionated recommendation and why, and then use AskUserQuestion. Also NUMBER issues and then give LETTERS for options and when using AskUserQuestion make sure each option clearly labels the issue NUMBER and option LETTER so the user doesn't get confused. Make the recommended option always the 1st option.



## Project Overview

PyxTrackers is a high-performance Cython reimplementation of three multi-object tracking algorithms: **SORT**, **ByteTrack**, and **OC-SORT**.
The goal is numerical equivalence with the Python originals while achieving significant speedups.
It is a standalone pip-installable package under the `pyxtrackers` namespace.

## Package Name & Namespace

```python
from pyxtrackers import Sort, BYTETracker, OCSort

# Or directly
from pyxtrackers.sort import Sort
from pyxtrackers.bytetrack import BYTETracker
from pyxtrackers.ocsort import OCSort
```

## Development Workflow

The build backend is setuptools. Use pip for installation and dependency management. No third-party project managers (uv, poetry, etc.) — standard pip + venv is sufficient for this project's scope.

### First-Time Setup

```bash
python -m venv .venv                     # Create virtual environment
source .venv/bin/activate                # Activate (Linux/macOS)
# .venv\Scripts\activate                 # Activate (Windows)
pip install -e ".[dev]"                  # Install package + dev dependencies
python setup.py build_ext --inplace      # Compile Cython extensions (.pyx → .so)
```

### Common Commands

```bash
pytest tests/ -v                         # Run tests
python setup.py build_ext --inplace      # Rebuild after changing .pyx files
```

### Why editable install (`-e`) + explicit `build_ext --inplace`

We use `pip install -e .` (editable) for development instead of `pip install .` (non-editable):

- **Editable (`-e`)**: Python imports resolve directly from the source tree. Editing a `.py` file takes effect immediately without reinstalling. But `.so` files (compiled Cython) may not be placed in the source tree depending on the setuptools version, so we run `python setup.py build_ext --inplace` explicitly to guarantee `.so` files land next to the `.pyx` files.
- **Non-editable**: Copies everything to `site-packages/`. Every change (even `.py`) requires a full `pip install .`, which recompiles all Cython extensions from scratch. Too slow for iterative development.

The dev workflow is:
- Edited `.py` → do nothing, changes are picked up immediately
- Edited `.pyx` → run `python setup.py build_ext --inplace` (incremental, only recompiles changed files)

For end users, `pip install pyxtrackers` or `pip install .` (non-editable) always places `.so` files correctly — the editable quirk only affects development.

### Dependency management

Build dependencies (Cython, numpy, setuptools-scm) are declared once in `[build-system].requires` in `pyproject.toml`. pip installs them automatically during `pip install -e .` or `pip install .`. They should NOT be duplicated in `[project.optional-dependencies].dev` — the dev extra only contains tools that are genuinely dev-only (pytest, filterpy, lap, pyinstaller, altair, vl-convert-python).

- Runtime dependencies: edit `[project].dependencies` in `pyproject.toml`
- Build dependencies: edit `[build-system].requires` in `pyproject.toml`
- Dev-only dependencies: edit `[project.optional-dependencies].dev` in `pyproject.toml`
- Then run `pip install -e ".[dev]"` to apply changes

### Versioning

Version is managed by `setuptools-scm` with CalVer (`YYYY.M.D`) derived from git tags automatically.

```bash
git tag v2026.2.23                       # Tag a release (first of the day)
git tag v2026.2.23.1                     # Second release same day
git push origin v2026.2.23              # Push tag → triggers CI release pipeline
```

Untagged commits get dev versions like `2026.2.24.0.dev3`. The version is available at runtime via `pyxtrackers.__version__`.

Note: PEP 440 normalizes leading zeros, so the version is `2026.2.23` not `2026.02.23`.

### Releasing

1. Tag the commit: `git tag v2026.2.23 && git push origin v2026.2.23`
2. GitHub Actions builds sdist + wheels for all platforms and publishes to PyPI
3. GitHub Release is created with wheels + standalone CLI binaries attached
4. For conda-forge: submit `conda-recipe/meta.yaml` to `conda-forge/staged-recipes` (first release only; subsequent releases are auto-detected)

### Build Flags

`setup.py` auto-detects the platform and build context:
- **Source install** (`pip install .`): Uses `-march=native` for max performance on the user's CPU
- **Binary wheels** (cibuildwheel/conda-build): Uses portable flags only (no `-march=native`)
- **Windows** (MSVC): Uses `/O2 /fp:fast` instead of GCC/Clang flags

### Managing Dependencies

- Runtime dependencies: edit `[project].dependencies` in `pyproject.toml`
- Dev dependencies: edit `[project.optional-dependencies].dev` in `pyproject.toml`
- Then run `pip install -e ".[dev]"` to apply changes

## Running Tests

Tests compare Python reference implementations against Cython implementations on shared detection data.

```bash
pytest                                   # All tests
pytest tests/ -v                         # Verbose
pytest tests/test_sort_comparison.py -v  # Single file
```

Numerical comparison tolerance: 1e-6 pixels.

## Architecture

### Three-Layer Design

1. **`pyxtrackers/`** — Cython reimplementations (the installable package)
2. **`references/`** — Pure Python reference implementations (for testing, not installed)
3. **`tests/`** — Comparison tests that run both implementations on identical input and verify equivalence

### Installable Package Structure

```
pyxtrackers/
├── __init__.py              # Re-exports Sort, BYTETracker, OCSort
├── sort/
│   ├── __init__.py          # Re-exports Sort
│   ├── sort.pyx             # Main SORT tracker
│   ├── sort.pyi             # Type stubs
│   ├── kalman_filter.pyx    # 7D Kalman filter
│   └── kalman_filter.pxd    # C-level interface
├── bytetrack/
│   ├── __init__.py          # Re-exports BYTETracker
│   ├── bytetrack.pyx        # Main ByteTrack tracker
│   ├── kalman_filter.pyx    # 8D Kalman filter
│   ├── kalman_filter.pxd
│   ├── matching.pyx         # IOU + linear assignment
│   └── matching.pxd
└── ocsort/
    ├── __init__.py           # Re-exports OCSort (wrapper)
    ├── ocsort.pyx            # Main OC-SORT tracker
    ├── ocsort.pyi            # Type stubs
    ├── ocsort_wrapper.py     # Python wrapper for SORT-compatible interface
    ├── kalman_filter.pyx     # 7D Kalman filter with freeze/unfreeze
    ├── kalman_filter.pxd
    ├── association.pyx       # IOU + linear assignment
    └── association.pxd
```

`.pxd` files expose `cdef` functions/structs for cross-module `cimport` without Python overhead.

### Build System

- **`pyproject.toml`** — Package metadata (PEP 621), build deps, pip config
- **`setup.py`** — Cython extension definitions (setuptools build backend)
- **`vendor/lapjv/`** — Vendored LAPJV C++ linear assignment solver

### Tracker Specifics

| Tracker | State Dim | State Vector | Notes |
|---------|-----------|-------------|-------|
| SORT | 7D | [x, y, s, r, vx, vy, vs] | Constant velocity model |
| ByteTrack | 8D | [x, y, a, h, vx, vy, va, vh] | Two-stage association (high/low confidence) |
| OC-SORT | 7D | [x, y, s, r, vx, vy, vs] | Freeze/unfreeze for occlusion handling |

### Key Cython Patterns

- **C structs over Python classes** for track state (e.g., `STrack` in ByteTrack) — no Python object overhead
- **Flat arrays for matrices** (e.g., `double covariance[64]` for 8x8) — cache-friendly layout
- **`nogil` sections** for GIL-free computation in hot paths
- **`malloc`/`free`** for manual memory management; `libcpp.vector` for dynamic collections
- **`cdef` functions** for C-only internal calls; `def` functions as Python-callable wrappers
- **View classes** (e.g., `STrackView`) provide non-owning Python access to C structs without copies

### Bounding Box Conventions

Three representations used throughout:
- **tlbr**: `[x1, y1, x2, y2]` — for IOU calculation
- **tlwh**: `[x, y, w, h]` — for storage
- **xyah**: `[cx, cy, aspect_ratio, h]` — for Kalman filter observation

ByteTrack IOU uses PASCAL VOC formula (+1 to dimensions); SORT/OC-SORT use standard formula.

### External Dependencies

- **LAPJV** (`vendor/lapjv/lapjv.cpp`): Vendored C++ linear assignment solver, linked at compile time
- **NumPy**: Runtime dependency; C headers used for compilation
- **Cython**: Build-time requirement
- **setuptools**: Build backend

### Track Lifecycle (all trackers)

```
New → Tracked → Lost → Removed
       ↑          |
       └──────────┘  (re-activated on match)
```
