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

PyXTrackers is a high-performance Cython reimplementation of three multi-object tracking algorithms: **SORT**, **ByteTrack**, and **OC-SORT**. The goal is numerical equivalence with the Python originals while achieving significant speedups (5.9x+ demonstrated for ByteTrack).

This repo is extracted from a larger project called **polyis**. The `setup.py`, tests, and import paths still reference the `polyis` namespace (e.g., `polyis.tracker.bytetrack.cython.bytetrack`). The Cython source files live in `trackers/` here but are compiled as part of the polyis package.

## Build Commands

```bash
# Build all Cython extensions (builds inplace by default)
python setup.py build_ext
```

## Running Tests

Tests compare Python reference implementations against Cython implementations on shared detection data.

```bash
python -m pytest tests/ -v  # all tests
```

Numerical comparison tolerance: 1e-6 pixels.

## Architecture

### Three-Layer Design

1. **`references/`** — Pure Python reference implementations (the originals being reimplemented)
2. **`trackers/`** — Cython reimplementations organized by tracker algorithm
3. **`tests/`** — Comparison tests that run both implementations on identical input and verify equivalence

### Per-Tracker Module Structure

Each tracker in `trackers/` follows the same pattern:
- **`<tracker>.pyx`** — Main tracker logic (track management, update loop)
- **`kalman_filter.pyx` / `.pxd`** — Kalman filter with C-level interface via `.pxd`
- **`matching.pyx` or `association.pyx` / `.pxd`** — IOU computation + linear assignment (LAPJV)
- Optional: `.pyi` type stubs, Python wrapper files

`.pxd` files expose `cdef` functions/structs for cross-module `cimport` without Python overhead.

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

- **LAPJV** (`modules/lap/_lapjv_cpp/lapjv.cpp`): C++ linear assignment solver linked at compile time
- **NumPy**: Array interfaces and C headers for compilation
- **Cython**: Build-time requirement

### Track Lifecycle (all trackers)

```
New → Tracked → Lost → Removed
       ↑          |
       └──────────┘  (re-activated on match)
```
