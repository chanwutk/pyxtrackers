from setuptools import setup, Extension, Command
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy
import os
import glob
import multiprocessing as mp


ARGS = ["-O3", "-ffast-math", "-march=native", "-mtune=native", "-finline-functions"]
MACROS: list[tuple[str, str | None]] = [("NPY_NO_DEPRECATED_API", "NPY_2_3_API_VERSION")]
LAPJV_DIR = os.path.join("vendor", "lapjv")


extensions = [
    # SORT
    Extension(
        "pyxtrackers.sort.kalman_filter",
        ["pyxtrackers/sort/kalman_filter.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=MACROS,
        extra_compile_args=ARGS,
    ),
    Extension(
        "pyxtrackers.sort.sort",
        [
            "pyxtrackers/sort/sort.pyx",
            os.path.join(LAPJV_DIR, "lapjv.cpp"),
        ],
        include_dirs=[LAPJV_DIR, numpy.get_include()],
        define_macros=MACROS,
        extra_compile_args=ARGS + ["-std=c++11"],
        language="c++",
    ),
    # OC-SORT
    Extension(
        "pyxtrackers.ocsort.kalman_filter",
        ["pyxtrackers/ocsort/kalman_filter.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=MACROS,
        extra_compile_args=ARGS,
    ),
    Extension(
        "pyxtrackers.ocsort.association",
        [
            "pyxtrackers/ocsort/association.pyx",
            os.path.join(LAPJV_DIR, "lapjv.cpp"),
        ],
        include_dirs=[LAPJV_DIR, numpy.get_include()],
        define_macros=MACROS,
        extra_compile_args=ARGS + ["-std=c++11"],
        language="c++",
    ),
    Extension(
        "pyxtrackers.ocsort.ocsort",
        ["pyxtrackers/ocsort/ocsort.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=MACROS,
        extra_compile_args=ARGS + ["-std=c++11"],
        language="c++",
    ),
    # ByteTrack
    Extension(
        "pyxtrackers.bytetrack.kalman_filter",
        ["pyxtrackers/bytetrack/kalman_filter.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=MACROS,
        extra_compile_args=ARGS,
    ),
    Extension(
        "pyxtrackers.bytetrack.matching",
        [
            "pyxtrackers/bytetrack/matching.pyx",
            os.path.join(LAPJV_DIR, "lapjv.cpp"),
        ],
        include_dirs=[LAPJV_DIR, numpy.get_include()],
        define_macros=MACROS,
        extra_compile_args=ARGS + ["-std=c++11"],
        language="c++",
    ),
    Extension(
        "pyxtrackers.bytetrack.bytetrack",
        ["pyxtrackers/bytetrack/bytetrack.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=MACROS,
        extra_compile_args=ARGS + ["-std=c++11"],
        language="c++",
    ),
]


class CleanCommand(Command):
    """Custom clean command to remove build artifacts."""

    description = "Remove build artifacts (*.c, *.cpp, *.html, *.so files)"
    user_options = []

    c_patterns = [
        "pyxtrackers/**/*.c",
        "pyxtrackers/**/*.cpp",
    ]
    so_patterns = [
        "pyxtrackers/**/*.so",
    ]
    html_patterns = [
        "pyxtrackers/**/*.html",
    ]

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """Execute the clean command."""
        removed_count = 0
        for pattern in self.c_patterns + self.so_patterns + self.html_patterns:
            for filepath in glob.glob(pattern, recursive=True):
                try:
                    os.remove(filepath)
                    print(f"Removed: {filepath}")
                    removed_count += 1
                except OSError as e:
                    print(f"Error removing {filepath}: {e}")

        print(f"\nCleaned {removed_count} file(s)")


class CleanAnnotateCommand(CleanCommand):
    """Custom clean command to remove only annotation artifacts (*.c, *.cpp, *.html files)."""

    description = "Remove annotation artifacts (*.c, *.cpp, *.html files only)"
    so_patterns = []


class BuildExt(build_ext):
    """Custom build_ext command that defaults to --inplace."""

    def initialize_options(self):
        super().initialize_options()
        self.inplace = True


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
    cmdclass={
        "clean": CleanCommand,
        "clean_annotate": CleanAnnotateCommand,
        "build_ext": BuildExt,
    },
    zip_safe=False,
)
