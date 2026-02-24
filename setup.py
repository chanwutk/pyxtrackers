from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os
import sys
import platform
import multiprocessing as mp


MACROS: list[tuple[str, str | None]] = [("NPY_NO_DEPRECATED_API", "NPY_2_3_API_VERSION")]
LAPJV_DIR = os.path.join("vendor", "lapjv")

# Portable builds (cibuildwheel, conda-build) should not use arch-specific flags
# that would make the binary incompatible with other CPUs.
_IS_PORTABLE = (
    os.environ.get("CIBUILDWHEEL") == "1"
    or os.environ.get("CONDA_BUILD") == "1"
)


def get_compile_args(is_cpp=False):
    """Return compiler flags appropriate for the current platform and build mode."""
    if sys.platform == "win32":
        args = ["/O2", "/fp:fast"]
        if is_cpp:
            args.append("/std:c++14")
    else:
        args = ["-O3", "-ffast-math"]
        if not _IS_PORTABLE:
            # Source install: optimize for the user's CPU
            if sys.platform == "darwin" and platform.machine() == "arm64":
                # Apple Clang on arm64 doesn't reliably support -march=native
                args.append("-mcpu=apple-m1")
            else:
                args.extend(["-march=native", "-mtune=native"])
        elif sys.platform == "darwin" and platform.machine() == "arm64":
            # Portable wheel on Apple Silicon: safe baseline for all M-series chips
            args.append("-mcpu=apple-m1")
        if is_cpp:
            args.append("-std=c++11")
    return args


extensions = [
    # SORT
    Extension(
        "pyxtrackers.sort.kalman_filter",
        ["pyxtrackers/sort/kalman_filter.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=MACROS,
        extra_compile_args=get_compile_args(),
    ),
    Extension(
        "pyxtrackers.sort.sort",
        [
            "pyxtrackers/sort/sort.pyx",
            os.path.join(LAPJV_DIR, "lapjv.cpp"),
        ],
        include_dirs=[LAPJV_DIR, numpy.get_include()],
        define_macros=MACROS,
        extra_compile_args=get_compile_args(is_cpp=True),
        language="c++",
    ),
    # OC-SORT
    Extension(
        "pyxtrackers.ocsort.kalman_filter",
        ["pyxtrackers/ocsort/kalman_filter.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=MACROS,
        extra_compile_args=get_compile_args(),
    ),
    Extension(
        "pyxtrackers.ocsort.association",
        [
            "pyxtrackers/ocsort/association.pyx",
            os.path.join(LAPJV_DIR, "lapjv.cpp"),
        ],
        include_dirs=[LAPJV_DIR, numpy.get_include()],
        define_macros=MACROS,
        extra_compile_args=get_compile_args(is_cpp=True),
        language="c++",
    ),
    Extension(
        "pyxtrackers.ocsort.ocsort",
        ["pyxtrackers/ocsort/ocsort.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=MACROS,
        extra_compile_args=get_compile_args(is_cpp=True),
        language="c++",
    ),
    # Utils
    Extension(
        "pyxtrackers.utils.scale",
        ["pyxtrackers/utils/scale.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=MACROS,
        extra_compile_args=get_compile_args(),
    ),
    # ByteTrack
    Extension(
        "pyxtrackers.bytetrack.kalman_filter",
        ["pyxtrackers/bytetrack/kalman_filter.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=MACROS,
        extra_compile_args=get_compile_args(),
    ),
    Extension(
        "pyxtrackers.bytetrack.matching",
        [
            "pyxtrackers/bytetrack/matching.pyx",
            os.path.join(LAPJV_DIR, "lapjv.cpp"),
        ],
        include_dirs=[LAPJV_DIR, numpy.get_include()],
        define_macros=MACROS,
        extra_compile_args=get_compile_args(is_cpp=True),
        language="c++",
    ),
    Extension(
        "pyxtrackers.bytetrack.bytetrack",
        ["pyxtrackers/bytetrack/bytetrack.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=MACROS,
        extra_compile_args=get_compile_args(is_cpp=True),
        language="c++",
    ),
]

# Test-only reference extension: only compile for local development, not for distribution
if not _IS_PORTABLE:
    extensions.insert(0, Extension(
        "references.bytetrack.cython_bbox",
        ["references/bytetrack/cython_bbox.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=MACROS,
        extra_compile_args=get_compile_args(),
    ))


ext_modules = cythonize(
    extensions,
    nthreads=mp.cpu_count() // 2,
    compiler_directives={
        "language_level": 3,
        "boundscheck": False,
        "wraparound": False,
        "initializedcheck": False,
        "nonecheck": False,
        "freethreading_compatible": True,
        "subinterpreters_compatible": "own_gil",
        "overflowcheck": False,
        "overflowcheck.fold": False,
        "embedsignature": False,
        "cdivision": True,
        "cpow": True,
        "optimize.use_switch": True,
        "optimize.unpack_method_calls": True,
        "warn.undeclared": False,
        "warn.unreachable": False,
        "warn.maybe_uninitialized": False,
        "warn.unused": False,
        "warn.unused_arg": False,
        "warn.unused_result": False,
        "warn.multiple_declarators": False,
        "infer_types": True,
        "infer_types.verbose": False,
        "profile": False,
        "linetrace": False,
        "emit_code_comments": False,
        "annotation_typing": False,
        "c_string_type": "str",
        "c_string_encoding": "ascii",
        "type_version_tag": True,
        "unraisable_tracebacks": False,
        "iterable_coroutine": True,
        "fast_gil": True,
    },
    annotate=True,
)

setup(
    ext_modules=ext_modules,
    zip_safe=False,
)
