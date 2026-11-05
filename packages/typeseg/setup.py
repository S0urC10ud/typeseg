import os
import sys

from setuptools import Extension, setup


compile_args = (
    ["/std:c++17", "/O2", "/fp:precise"]
    if sys.platform == "win32"
    else ["-std=c++17", "-O3", "-ffp-contract=off"]
)

setup(
    ext_modules=[
        Extension(
            "typeseg._scan_cpu",
            sources=["native/scan_cpu.cpp"],
            include_dirs=["native/onnxruntime"],
            define_macros=[("Py_LIMITED_API", "0x03090000")],
            extra_compile_args=compile_args,
            language="c++",
            py_limited_api=True,
            # Release builders must fail rather than publish a wheel that
            # silently lacks the accelerated scan. Local source installs can
            # still use the original ONNX graph without a compiler.
            optional=os.environ.get("TYPESEG_REQUIRE_NATIVE") != "1",
        )
    ],
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
