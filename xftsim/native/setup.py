"""
Build script for the grg_recomb_native extension.

Mirrors the CMakeExtension + CMakeBuild pattern used by grgl/setup.py so the
upstream migration is structurally a no-op: when the C++ files move into
grgl/, the binding registration folds into grgl's _grgl.cpp and this setup.py
disappears.
"""

import copy
import os
import subprocess
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


C_MODULE_NAME = "grg_recomb_native._grg_recomb_native"
THISDIR = os.path.realpath(os.path.dirname(__file__))


class CMakeExtension(Extension):
    def __init__(self, name, cmake_lists_dir="."):
        Extension.__init__(self, name, sources=[])
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class CMakeBuild(build_ext):
    def get_source_files(self):
        sources = ["CMakeLists.txt"]
        for root, _dirs, files in os.walk(os.path.join(THISDIR, "src")):
            for f in files:
                sources.append(os.path.join(root, f))
        for root, _dirs, files in os.walk(os.path.join(THISDIR, "include")):
            for f in files:
                sources.append(os.path.join(root, f))
        return sources

    def build_extensions(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as exc:
            raise RuntimeError("cmake not found on PATH") from exc

        for ext in self.extensions:
            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
            cmake_args = [
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(extdir),
                "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE={}".format(extdir),
                "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={}".format(self.build_temp),
                "-DPYTHON_EXECUTABLE={}".format(sys.executable),
            ]
            grgl_root = os.environ.get("GRGL_ROOT")
            if grgl_root:
                cmake_args.append("-DGRGL_ROOT={}".format(grgl_root))

            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)

            # grgl's CMakeLists invokes the BGEN third-party project which can
            # fail the CMake policy check on older configs; preserve the same
            # workaround grgl/setup.py uses.
            env = copy.deepcopy(os.environ)
            env["CMAKE_POLICY_VERSION_MINIMUM"] = "3.5"

            subprocess.check_call(
                ["cmake", ext.cmake_lists_dir] + cmake_args,
                cwd=self.build_temp,
                stdout=sys.stdout,
                env=env,
            )
            subprocess.check_call(
                ["cmake", "--build", ".", "--config", "Release", "--parallel"],
                cwd=self.build_temp,
                stdout=sys.stdout,
                env=env,
            )


setup(
    name="grg_recomb_native",
    version="0.1.0",
    description="Native C++ recombination algorithms for pygrgl",
    packages=find_packages(),
    ext_modules=[CMakeExtension(C_MODULE_NAME)],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.9",
)
