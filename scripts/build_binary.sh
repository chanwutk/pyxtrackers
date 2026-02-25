#!/usr/bin/env bash
set -euo pipefail

# Build Cython extensions
uv run python setup.py build_ext

# Bundle into a single binary
uv run pyinstaller --onefile \
    --name pyxtrackers \
    --hidden-import pyxtrackers.sort.sort \
    --hidden-import pyxtrackers.sort.kalman_filter \
    --hidden-import pyxtrackers.bytetrack.bytetrack \
    --hidden-import pyxtrackers.bytetrack.kalman_filter \
    --hidden-import pyxtrackers.bytetrack.matching \
    --hidden-import pyxtrackers.ocsort.ocsort \
    --hidden-import pyxtrackers.ocsort.kalman_filter \
    --hidden-import pyxtrackers.ocsort.association \
    --hidden-import pyxtrackers.utils.scale \
    pyxtrackers/cli_launcher.py

echo "Binary built at: dist/pyxtrackers"
