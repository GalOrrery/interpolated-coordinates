0.1 (2021-02-02)
================

Initial release.


New Features
------------


interpolated_coordinates
^^^^^^^^^^^^^^^^^^^^^^^^

- InterpolatedBaseRepresentationOrDifferential
- InterpolatedRepresentation
- InterpolatedCartesianRepresentation
- InterpolatedDifferential
- InterpolatedCoordinateFrame
- InterpolatedSkyCoord


interpolated_coordinates.utils
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Unit-aware spline classes:

    + UnivariateSplinewithUnits
    + InterpolatedUnivariateSplinewithUnits
    + LSQUnivariateSplinewithUnits

- Generic Representations:

    + GenericRepresentation
    + GenericDifferential
    + Methods to make subclasses, given an Astropy Representation or Differential


Other Changes and Additions
---------------------------

- A configuration file with all options set to their defaults is now generated
