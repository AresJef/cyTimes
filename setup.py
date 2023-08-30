import os, numpy as np, platform
from setuptools import setup, Extension
from Cython.Build import cythonize

# Package name
__package__ = "cytimes"


# Create Extension
def extension(filename: str, include_np: bool, *extra_compile_args: str) -> Extension:
    # Extra arguments
    extra_args = list(extra_compile_args) if extra_compile_args else None
    # Name
    name: str = "%s.%s" % (__package__, filename.split(".")[0])
    source: str = os.path.join("src", __package__, filename)
    # Create extension
    if include_np:
        return Extension(
            name,
            sources=[source],
            extra_compile_args=extra_args,
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        )
    else:
        return Extension(name, sources=[source], extra_compile_args=extra_args)


# Build Extensions
if platform.system() == "Windows":
    extensions = [
        extension("cydatetime.py", True),
        extension("cymath.py", False),
        extension("cyparser.py", True),
        extension("cytime.pyx", False),
        extension("cytimedelta.py", True),
        extension("pydt.py", True),
        extension("pddt.py", True),
    ]
else:
    extensions = [
        extension("cydatetime.py", True, "-Wno-unreachable-code"),
        extension("cymath.py", False, "-Wno-unreachable-code"),
        extension("cyparser.py", True, "-Wno-unreachable-code"),
        extension("cytime.pyx", False, "-Wno-unreachable-code"),
        extension("cytimedelta.py", True, "-Wno-unreachable-code"),
        extension("pydt.py", True, "-Wno-unreachable-code"),
        extension("pddt.py", True, "-Wno-unreachable-code"),
    ]

# Build
setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
        annotate=True,
    ),
)
