# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Interpolated Coordinates, Representations, and SkyCoords.

Astropy `coordinate <astropy.coordinates.SkyCoord>` objects are collections of
points. This module provides wrappers to interpolate each dimension of a
coordinate object with an affine parameter.

For all the following examples we assume these imports:

    >>> import astropy.units as u
    >>> import astropy.coordinates as coord
    >>> import numpy as np
    >>> import interpolated_coordinates as icoord

We will start with interpolating Representation object.

    >>> num = 40
    >>> affine = np.linspace(0, 10, num=num) * u.Myr
    >>> rep = coord.CartesianRepresentation(
    ...     x=np.linspace(0, 1, num=num) * u.kpc,
    ...     y=np.linspace(1, 2, num=num) * u.kpc,
    ...     z=np.linspace(2, 3, num=num) * u.kpc,
    ...     differentials=coord.CartesianDifferential(
    ...         d_x=np.linspace(3, 4, num=num) * (u.km / u.s),
    ...         d_y=np.linspace(4, 5, num=num) * (u.km / u.s),
    ...         d_z=np.linspace(5, 6, num=num) * (u.km / u.s)))
    >>> irep = icoord.InterpolatedRepresentation(rep, affine)
    >>> irep[:4]
    <InterpolatedCartesianRepresentation (affine| x, y, z) in Myr| kpc
        [(0.        , 0.        , 1.        , 2.        ),
         (0.25641026, 0.02564103, 1.02564103, 2.02564103),
         (0.51282051, 0.05128205, 1.05128205, 2.05128205),
         (0.76923077, 0.07692308, 1.07692308, 2.07692308)]
     (has differentials w.r.t.: 's')>

Interpolation means we can get the coordinate (representation) at any point
supported by the affine parameter. For example, the Cartesian coordinate at some
arbitrary value, say ``affine=4.873 * u.Myr``, is

    >>> irep(4.873 * u.Myr)
    <CartesianRepresentation (x, y, z) in kpc
        (0.4873, 1.4873, 2.4873)
     (has differentials w.r.t.: 's')>

The interpolation can be evaluated on a scalar or any shaped |Quantity| array,
returning a Representation with the same shape.

This interpolation machinery is built on top of Astropy's Representation class
and supports all the expected operations, like changing representations, while
maintaining the interpolation.

    >>> irep.represent_as(coord.SphericalRepresentation)[:4]
    <InterpolatedSphericalRepresentation (affine| lon, lat, distance) in ...
        [(0.        , 1.57079633, 1.10714872, 2.23606798),
         (0.25641026, 1.54580153, 1.10197234, 2.27064276),
         (0.51282051, 1.52205448, 1.09671629, 2.30555457),
         (0.76923077, 1.49948886, 1.09140331, 2.34078832)]>

Also supported are some of :mod:`~scipy` interpolation methods. In particular,
we can differentiate the interpolated coordinates with respect to the affine
parameter.

    >>> irep.derivative()[:4]
    <InterpolatedCartesianDifferential (affine| d_x, d_y, d_z) in ...
        [(0.        , 0.1, 0.1, 0.1), (0.25641026, 0.1, 0.1, 0.1),
         (0.51282051, 0.1, 0.1, 0.1), (0.76923077, 0.1, 0.1, 0.1)]>

Note that the result is an interpolated Differential class. Higher-order
derivatives can also be constructed, but they do not have a corresponding class
in Astropy, so a "Generic" class is constructed.

    >>> irep.derivative(n=2)[:4]
    <InterpolatedGenericCartesian2ndDifferential (affine| d_x, d_y, d_z) in ...
        [(0.        , -5.41233725e-16,  3.35564909e-15, -9.45535317e-14),
         (0.25641026,  1.80411242e-17, -2.88657986e-16, -1.91326122e-14),
         (0.51282051,  5.77315973e-16, -3.93296506e-15,  5.62883073e-14),
         (0.76923077, -8.65973959e-16,  5.89944760e-15, -5.06594766e-14)]>

Care should be taken not to change representations for these higher-order
derivatives. The Astropy machinery allows them to be transformed, but the
transformation is often incorrect.


Representations are all well and good, but what about coordinate frames? The
interpolated representations can be used the same as Astropy's, including in a
|Frame|.

    >>> frame = coord.ICRS(irep)
    >>> frame[:1]
    <ICRS Coordinate: (ra, dec, distance) in (deg, deg, kpc)
        [(90., 63.43494882, 2.23606798)]
     (pm_ra_cosdec, pm_dec, radial_velocity) in (mas / yr, mas / yr, km / s)
        [(-0.28301849, -0.12656972, 6.26099034)]>

The underlying representation is still interpolated, and the interpolation is
even kept when transforming frames.

    >>> frame = frame.transform_to(coord.Galactic())
    >>> frame.data[:4]
    <InterpolatedCartesianRepresentation (affine| x, y, z) in Myr| kpc
        [(0.        , -1.8411072 , 1.04913465, 0.71389129),
         (0.25641026, -1.87731612, 1.06955162, 0.69825645),
         (0.51282051, -1.91352503, 1.08996859, 0.68262162),
         (0.76923077, -1.94973395, 1.11038556, 0.66698678)]
     (has differentials w.r.t.: 's')>

For deeper integration and access to interpolation methods the
``InterpolatedCoordinateFrame`` can wrap any ``CoordinateFame``, whether or not
it contains an interpolated representation.

    >>> iframe = icoord.InterpolatedCoordinateFrame(frame)
    >>> iframe[:4]
    <InterpolatedGalactic Coordinate: (affine| l, b, distance) in ...
        [(0.        , 150.32382371, 18.61829304, 2.23606798),
         (0.25641026, 150.32880684, 17.90952972, 2.27064276),
         (0.51282051, 150.33360184, 17.22212858, 2.30555457),
         (0.76923077, 150.33821918, 16.55532737, 2.34078832)]
     (affine| pm_l, pm_b, radial_velocity) in (Myr| mas / yr, mas / yr, km / s)
        [(0.        , 0.00218867, -0.31002428, 6.26099034),
         (0.25641026, 0.00210526, -0.30065482, 6.33590983),
         (0.51282051, 0.00202654, -0.29161849, 6.40935614),
         (0.76923077, 0.00195215, -0.28290567, 6.48140523)]>

When wrapping an un-interpolated coordinate, the affine parameter is required.

    >>> frame = coord.ICRS(rep)  # no interp
    >>> iframe = icoord.InterpolatedCoordinateFrame(frame, affine=affine)
    >>> iframe[:4]
    <InterpolatedICRS Coordinate: (affine| ra, dec, distance) in ...
        [(0.        , 90.        , 63.43494882, 2.23606798),
         (0.25641026, 88.56790382, 63.13836438, 2.27064276),
         (0.51282051, 87.20729763, 62.83721465, 2.30555457),
         (0.76923077, 85.91438322, 62.53280357, 2.34078832)]
     (affine| pm_ra, pm_dec, radial_velocity) in ...
        [(0.        , -0.63284858, -0.12656972, 6.26099034),
         (0.25641026, -0.60122591, -0.12884151, 6.33590983),
         (0.51282051, -0.57125382, -0.13051534, 6.40935614),
         (0.76923077, -0.54290056, -0.13166259, 6.48140523)]>

Just as for interpolated representations, interpolated frames can be evaluated,
differentiated, etc.

    >>> iframe(4.873 * u.Myr)
    <ICRS Coordinate: (ra, dec, distance) in (deg, deg, kpc)
        (71.8590987, 57.82047953, 2.93873848)
     (pm_ra_cosdec, pm_dec, radial_velocity) in (mas / yr, mas / yr, km / s)
        (-0.13759357, -0.1152677, 7.49365212)>

There are also interpolated |SkyCoord|. This is actually a direct subclass of
SkyCoord, not a proxy class like the interpolated representations and coordinate
frame. As such, ``InterpolatedSkyCoord`` can be instantiated in all the normal
ways, except that it requires the kwarg ``affine``. The only exception is if
SkyCoord is wrapping an interpolated CoordinateFrame.

    >>> isc = icoord.InterpolatedSkyCoord(
    ...         [1, 2, 3, 4], [-30, 45, 8, 16],
    ...         frame="icrs", unit="deg",
    ...         affine=affine[:4])
    >>> isc
    <InterpolatedSkyCoord (ICRS): (affine| ra, dec) in Myr| deg
        [(0.        , 1., -30.), (0.25641026, 2.,  45.),
         (0.51282051, 3.,   8.), (0.76923077, 4.,  16.)]>

"""

# ----------------------------------------------------------------------------
from .frame import *  # noqa: F403
from .representation import *  # noqa: F403
