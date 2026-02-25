"""Pytest configuration for pyxtrackers tests.

Adds:
- ``--visualize`` flag: controls which scale tests run (all vs subset).
- ``--cli-binary`` option: path to a PyInstaller binary for binary-CLI tests.

After tests finish, perf records are saved to ``test-results/perf.jsonl``.
Use ``scripts/visualize.py`` to generate charts from that file.
"""

import json
import os


def pytest_addoption(parser):
    parser.addoption(
        "--visualize",
        action="store_true",
        default=False,
        help="Run full scalability benchmarks (all scales)",
    )
    parser.addoption(
        "--cli-binary",
        default=None,
        help="Path to PyInstaller binary for binary-CLI tests (default: skip binary tests)",
    )


def pytest_collection_modifyitems(config, items):
    deselected = []
    selected = []
    if config.getoption("--visualize"):
        for item in items:
            if "tests/test_trackers.py" not in item.nodeid:
                selected.append(item)
                continue
            if "len4" in item.nodeid:
                selected.append(item)
            else:
                deselected.append(item)
    else:
        for item in items:
            if "tests/test_trackers.py" not in item.nodeid:
                selected.append(item)
                continue
            if "s1" in item.nodeid and "len1" in item.nodeid:
                selected.append(item)
            else:
                deselected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


def pytest_sessionfinish(session, exitstatus):
    from tests.test_trackers import _PERF_RECORDS

    if not _PERF_RECORDS:
        return

    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "test-results"
    )
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "perf.jsonl")

    with open(output_path, "w") as f:
        for record in _PERF_RECORDS:
            f.write(json.dumps(record) + "\n")

    print(f"\nPerf results saved to {output_path}")
    print(f"Generate charts: python scripts/visualize.py")
