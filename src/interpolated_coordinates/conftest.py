"""Configure Test Suite.

This file is used to configure the behavior of pytest when using the Astropy
test infrastructure. It needs to live inside the package in order for it to
get picked up when running the tests inside an interpreter using
`interpolated_coordinates.test()`.

"""

import os
import pathlib
from typing import Any

import pytest
from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS

##############################################################################
# CODE
##############################################################################


def pytest_configure(config: Any) -> None:
    """Configure :mod:`pytest` with :mod:`astropy`.

    Parameters
    ----------
    config : pytest configuration
        Configuration for :mod:`pytest`.
    """
    config.option.astropy_header = True

    # Customize the following lines to add/remove entries from the list of
    # packages for which version numbers are displayed when running the tests.
    PYTEST_HEADER_MODULES.pop("Pandas", None)
    PYTEST_HEADER_MODULES["scikit-image"] = "skimage"

    from . import __version__

    packagename = pathlib.Path(__file__).resolve().parent.name
    TESTED_VERSIONS[packagename] = __version__


# This has to be in the root dir or it will not display in CI.
def pytest_report_header(config: Any) -> str:  # noqa: ARG001
    """Add extra info to the :mod:`pytest` header."""
    # This gets added after the pytest-astropy-header output.
    return (
        f'ARCH_ON_CI: {os.environ.get("ARCH_ON_CI", "undefined")}\n'
        f'IS_CRON: {os.environ.get("IS_CRON", "undefined")}\n'
    )


# ------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _add_numpy(doctest_namespace: Any) -> None:
    """Add :mod:`numpy` to :mod:`pytest`.

    Parameters
    ----------
    doctest_namespace : namespace
        Namespace for doctests.
    """
    import numpy as np

    # add to namespace
    doctest_namespace["np"] = np


@pytest.fixture(scope="session", autouse=True)
def _add_astropy(doctest_namespace: Any) -> None:
    """Add :mod:`astropy` stuff to :mod:`pytest`.

    Parameters
    ----------
    doctest_namespace : namespace
        Namespace for doctests.
    """
    import astropy.coordinates as coord
    import astropy.units

    # add to namespace
    doctest_namespace["coord"] = coord
    doctest_namespace["u"] = astropy.units
