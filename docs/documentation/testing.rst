.. _interpolated_coordinates-test:

=================
Running the Tests
=================

The tests are written assuming they will be run with `pytest
<http://doc.pytest.org/>`_ using the Astropy `custom test runner
<http://docs.astropy.org/en/stable/development/testguide.html>`_. To set up a
Conda environment to run the full set of tests, install
``interpolated_coordinates`` or see the setup.cfg file for dependencies.
Once the dependencies are installed, you can run the tests two ways:

1. By importing ``interpolated_coordinates``::

    import interpolated_coordinates
    interpolated_coordinates.test()

2. By cloning the repository and running

::

    python setup.py test

Or, for pytest

::

    pytest

Or, for pytest in an isolated environment

::

    tox -e test
