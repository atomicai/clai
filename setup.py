"""Setups simple package."""

import io
import os
import pathlib

import pkg_resources
import setuptools
from setuptools import setup

extras_require = dict(
    format=["isort==5.10.1", "black==22.3.0", "autoflake==1.4"],
    test=[
        "pytest",
        "pytest-sugar",  # For nicer look and feel
        "pytest-faker",  # For faker generator fixture
        # For running only subset of tests for changed files
        # Currently, testmon doesn't seem to work with xdist.
        # https://github.com/tarpas/pytest-testmon/issues/42
        "pytest-testmon==1.1.0",
        "pytest-custom-exit-code",  # For `--suppress-no-test-exit-code` option
    ],
)
extras_require["dev"] = sum((extras_require[k] for k in ["format", "test"]), [])
extras_require["all"] = sum(extras_require.values(), [])

setup(
    name="clai",
    version="1.0.0",
    install_requires=list(
        map(str, pkg_resources.parse_requirements(io.open(str(pathlib.Path(os.getcwd()) / "requirements.txt"))))
    ),
    extras_require=extras_require,
    python_requires=">=3.7",
    packages=setuptools.find_packages(),
    include_package_data=True,
)
