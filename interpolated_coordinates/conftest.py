# -*- coding: utf-8 -*-
"""Configure Test Suite.

This file is used to configure the behavior of pytest when using the Astropy
test infrastructure. It needs to live inside the package in order for it to
get picked up when running the tests inside an interpreter using
`interpolated_coordinates.test()`.

"""

##############################################################################
# IMPORTS

# STDLIB
import os

# THIRD PARTY
import pytest
from astropy.version import version as astropy_version

try:
    # THIRD PARTY
    from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS

    ASTROPY_HEADER = True
except ImportError:
    ASTROPY_HEADER = False

##############################################################################
# CODE
##############################################################################


def pytest_configure(config) -> None:
    """Configure :mod:`pytest` with :mod:`astropy`.

    Parameters
    ----------
    config : pytest configuration
    """
    if ASTROPY_HEADER:

        config.option.astropy_header = True

        # Customize the following lines to add/remove entries from the list of
        # packages for which version numbers are displayed when running the tests.
        PYTEST_HEADER_MODULES.pop("Pandas", None)
        PYTEST_HEADER_MODULES["scikit-image"] = "skimage"

        # LOCAL
        from . import __version__

        packagename = os.path.basename(os.path.dirname(__file__))
        TESTED_VERSIONS[packagename] = __version__


# ------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def add_numpy(doctest_namespace) -> None:
    """Add :mod:`numpy` to :mod:`pytest`.

    Parameters
    ----------
    doctest_namespace : namespace

    """
    # THIRD PARTY
    import numpy

    # add to namespace
    doctest_namespace["np"] = numpy


@pytest.fixture(scope="session", autouse=True)
def add_astropy(doctest_namespace) -> None:
    """Add :mod:`astropy` stuff to :mod:`pytest`.

    Parameters
    ----------
    doctest_namespace : namespace

    """
    # THIRD PARTY
    import astropy.coordinates as coord
    import astropy.units

    # add to namespace
    doctest_namespace["coord"] = coord
    doctest_namespace["u"] = astropy.units
