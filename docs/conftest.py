# Licensed under a 3-clause BSD style license - see LICENSE.rst

# This file needs to be included here to make sure commands such
# as ``pytest docs/...`` works, since this
# will ignore the conftest.py file at the root of the repository
# and the one in astropy/conftest.py

import os
import pathlib
import tempfile

import pytest

# Make sure we use temporary directories for the config and cache
# so that the tests are insensitive to local configuration.

os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp("interpolated_coodinates_config")
os.environ["XDG_CACHE_HOME"] = tempfile.mkdtemp("interpolated_coodinates_cache")

(pathlib.Path(os.environ["XDG_CONFIG_HOME"]) / "interpolated_coodinates").mkdir()
(pathlib.Path(os.environ["XDG_CACHE_HOME"]) / "interpolated_coodinates").mkdir()

# Note that we don't need to change the environment variables back or remove
# them after testing, because they are only changed for the duration of the
# Python process, and this configuration only matters if running pytest
# directly, not from e.g. an IPython session.


@pytest.fixture(autouse=True)
def _docdir(request):
    """Run doctests in isolated tmpdir so outputs do not end up in repo."""
    # Trigger ONLY for doctestplus
    doctest_plugin = request.config.pluginmanager.getplugin("doctestplus")
    if isinstance(request.node.parent, doctest_plugin._doctest_textfile_item_cls):
        # Don't apply this fixture to io.rst.  It reads files and doesn't write
        if "io.rst" not in request.node.name:
            tmpdir = request.getfixturevalue("tmpdir")
            with tmpdir.as_cwd():
                yield
        else:
            yield
    else:
        yield
