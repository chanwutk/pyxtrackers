"""
PyXTrackers: High-performance Cython multi-object tracking algorithms.

Provides Cython implementations of SORT, ByteTrack, and OC-SORT.
"""

from importlib.metadata import version, PackageNotFoundError

from pyxtrackers.sort.sort import Sort
from pyxtrackers.bytetrack.bytetrack import BYTETracker
from pyxtrackers.ocsort import OCSort

try:
    __version__ = version("pyxtrackers")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["Sort", "BYTETracker", "OCSort", "__version__"]
