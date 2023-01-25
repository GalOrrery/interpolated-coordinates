Documentation
=============

.. container::

   |DOI| |PyPI| |Build Status| |Codecov| |astropy|


Do you ever want to interpolate an Astropy Representation, Coordinate Frame,
or SkyCoord? If yes, this might be a useful package.

.. toctree::
   :maxdepth: 1

   documentation/installation


********
Packages
********

|

.. toctree::
   :maxdepth: 1

   interpolated_coordinates/index.rst

.. toctree::
   :caption: Codes and Utilities:
   :name: codes_and_utilities
   :maxdepth: 1

   interpolated_coordinates/spline.rst
   interpolated_coordinates/generic_representation.rst

***********
Attribution
***********

.. container::

   |DOI| |License|

Copyright 2018- Nathaniel Starkman and collaborators.

``interpolated_coordinates`` is free software made available under a
modified BSD-3 License. For details see the `LICENSE <https://github.com/GalOrrery/interpolated-coordinates/blob/master/LICENSE.rst>`_ file.


If you make use of this code, please use the Zenodo DOI as a software citation
::

    @software{interpolated-coordinates:zenodo,
     author       = {Starkman, Nathaniel},
     title        = Interpolated-Coordinates,
     publisher    = {Zenodo},
     doi          = {TODO},
     url          = {TODO}
    }


***************
Project details
***************

.. toctree::
   :maxdepth: 1

   documentation/code_quality
   documentation/testing
   latest version changelog (v0.1) <whatsnew/0.1>
   whatsnew/index
   credits

..
  RST COMMANDS BELOW

.. BADGES

.. |astropy| image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
   :target: http://www.astropy.org/

.. |Build Status| image:: https://github.com/GalOrrery/interpolated-coordinates/workflows/CI/badge.svg
    :target: https://github.com/GalOrrery/interpolated-coordinates

.. |Documentation Status| image:: https://readthedocs.org/projects/interpolated-coordinates/badge/?version=latest
   :target: https://interpolated-coordinates.readthedocs.io/en/latest/?badge=latest

.. |DOI| image:: https://zenodo.org/badge/453202992.svg
   :target: https://zenodo.org/badge/latestdoi/453202992

.. |License| image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
   :target: https://opensource.org/licenses/BSD-3-Clause

.. |PyPI| image:: https://badge.fury.io/py/interpolated-coordinates.svg
   :target: https://badge.fury.io/py/interpolated-coordinates

.. |Milestones| image:: https://img.shields.io/github/milestones/open/GalOrrery/interpolated-coordinates?style=flat
   :alt: GitHub milestones

.. |Open Issues| image:: https://img.shields.io/github/issues-raw/GalOrrery/interpolated-coordinates?style=flat
   :alt: GitHub issues

.. |Last Commit| image:: https://img.shields.io/github/last-commit/GalOrrery/interpolated-coordinates/master?style=flat
   :alt: GitHub last commit (branch)

.. |Codecov| image:: https://codecov.io/gh/GalOrrery/interpolated-coordinates/branch/main/graph/badge.svg?token=LXhzIKtrVo
  :target: https://codecov.io/gh/GalOrrery/interpolated-coordinates
