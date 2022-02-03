Univariate Interpolations of Astropy Coordinates
================================================

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge

|
|

.. contents:: Table of Contents:


License
-------

This project is Copyright (c) Nathaniel Starkman and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the `Astropy package template <https://github.com/astropy/package-template>`_
which is licensed under the BSD 3-clause license. See the licenses folder for
more information.


Contributing
------------

We love contributions! ``interpolated-coordinates`` is open source,
built on open source, and we'd love to have you hang out in our community.



Installation
------------

``interpolated-coordinates`` can by installed from `PyPI <https://pypi.org/project/interpolated-coordinates/>`_.


.. code-block:: bash

    >>> pip install interpolated_coordinates

If you're reading this on `GitHub <https://github.com/GalOrrery/interpolated-coordinates/>`_ the code can be built from source:

.. code-block:: bash

    >>> python setup.py install

Also in development mode:

.. code-block:: bash

    >>> python setup.py develop



Use
===

.. |astropy| replace:: `scipy <https://docs.astropy.org/en/stable>`_
.. |scipy| replace:: `scipy <https://docs.scipy.org/doc/scipy/reference/>`_
.. |Quantity| replace:: `Quantity <https://docs.astropy.org/en/stable/api/astropy.units.Quantity.html>`_
.. |CartesianRepresentation| replace:: `CartesianRepresentation <https://docs.astropy.org/en/stable/api/astropy.coordinates.CartesianRepresentation.html>`_
.. |Frame| replace:: `Coordinate Frame <https://docs.astropy.org/en/stable/api/astropy.coordinates.BaseCoordinateFrame.html>`_
.. |SkyCoord| replace:: `SkyCoord <https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html>`_
.. |InterpolatedUnivariateSpline| replace:: `InterpolatedUnivariateSpline <https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.interpolate.InterpolatedUnivariateSpline.html>`_


Astropy |SkyCoord| objects are collections of points.

This module provides wrappers to interpolate each dimension of a coordinate object with an affine parameter.

For all the following examples we assume these imports:

    >>> import astropy.units as u
    >>> import astropy.coordinates as coord
    >>> import numpy as np
    >>> import interpolated_coordinates as icoord


Representations
---------------

We will start with interpolating `Representation <https://docs.astropy.org/en/stable/api/astropy.coordinates.BaseRepresentation.html>`_ objects.

    >>> npts = 40  # the number of points
    >>> rep = coord.CartesianRepresentation(
    ...     x=np.linspace(0, 1, num=npts) * u.kpc,
    ...     y=np.linspace(1, 2, num=npts) * u.kpc,
    ...     z=np.linspace(2, 3, num=npts) * u.kpc,
    ...     differentials=coord.CartesianDifferential(
    ...         d_x=np.linspace(3, 4, num=npts) * (u.km / u.s),
    ...         d_y=np.linspace(4, 5, num=npts) * (u.km / u.s),
    ...         d_z=np.linspace(5, 6, num=npts) * (u.km / u.s)))

Now that the a standard |CartesianRepresentation| is defined, we can interpolate each dimension against an affine parameter. The affine parameter can have any units: time, arc length, furlongs per steradian, etc. So long as the value (of the affine parameter) works with |InterpolatedUnivariateSpline|, it's AOK.

    >>> affine = np.linspace(0, 10, npts=npts) * u.Myr
    >>> irep = icoord.InterpolatedRepresentation(rep, affine)
    >>> irep[:4]
    <InterpolatedCartesianRepresentation (affine| x, y, z) in Myr| kpc
        [(0.        , 0.        , 1.        , 2.        ),
         (0.25641026, 0.02564103, 1.02564103, 2.02564103),
         (0.51282051, 0.05128205, 1.05128205, 2.05128205),
         (0.76923077, 0.07692308, 1.07692308, 2.07692308)]
     (has differentials w.r.t.: 's')>

Interpolation means we can get the coordinate (representation) at any point
supported by the affine parameter. For example, the Cartesian coordinate
at some arbitrary value, say ``affine=4.873 * u.Myr``, is

    >>> irep(4.873 * u.Myr)
    <CartesianRepresentation (x, y, z) in kpc
        (0.4873, 1.4873, 2.4873)
     (has differentials w.r.t.: 's')>

The interpolation can be evaluated on a scalar or any shaped |Quantity|
array, returning a Representation with the same shape.

This interpolation machinery is built on top of Astropy's Representation
class and supports all the expected operations, like `changing representations <https://docs.astropy.org/en/stable/api/astropy.coordinates.BaseRepresentation.html#astropy.coordinates.BaseRepresentation.represent_as>`_,
while maintaining the interpolation.

    >>> irep.represent_as(coord.SphericalRepresentation)[:4]
    <InterpolatedSphericalRepresentation (affine| lon, lat, distance) in ...
        [(0.        , 1.57079633, 1.10714872, 2.23606798),
         (0.25641026, 1.54580153, 1.10197234, 2.27064276),
         (0.51282051, 1.52205448, 1.09671629, 2.30555457),
         (0.76923077, 1.49948886, 1.09140331, 2.34078832)]>

Also supported are some of |scipy| interpolation methods. In particular,
we can differentiate the interpolated coordinates with respect to the affine
parameter.

    >>> irep.derivative(n=1)[:4]
    <InterpolatedCartesianDifferential (affine| d_x, d_y, d_z) in ...
        [(0.        , 0.1, 0.1, 0.1), (0.25641026, 0.1, 0.1, 0.1),
         (0.51282051, 0.1, 0.1, 0.1), (0.76923077, 0.1, 0.1, 0.1)]>

Note that the result is an interpolated `Differential <https://docs.astropy.org/en/stable/api/astropy.coordinates.BaseDifferential.html>`_ class. Higher-order
derivatives can also be constructed, but they do not have a corresponding
class in Astropy, so a "Generic" class is constructed.

    >>> irep.derivative(n=2)[:4]
    <InterpolatedGenericCartesian2ndDifferential (affine| d_x, d_y, d_z) in ...
        [(0.        , -5.41233725e-16,  3.35564909e-15, -9.45535317e-14),
         (0.25641026,  1.80411242e-17, -2.88657986e-16, -1.91326122e-14),
         (0.51282051,  5.77315973e-16, -3.93296506e-15,  5.62883073e-14),
         (0.76923077, -8.65973959e-16,  5.89944760e-15, -5.06594766e-14)]>

Care should be taken not to change representations for these higher-order
derivatives. The Astropy machinery allows them to be transformed, but
the transformation **is most likely incorrect**. *If you are interested in improving representations of higher order differentials please open PRs with improvements, both here and especially in Astropy*.


Coordinate Frames
-----------------

Representations are all well and good, but what about coordinate frames?
The interpolated representations can be used the same as Astropy's, including
in a |Frame|.

    >>> frame = coord.ICRS(irep)
    >>> frame[:1]
    <ICRS Coordinate: (ra, dec, distance) in (deg, deg, kpc)
        [(90., 63.43494882, 2.23606798)]
     (pm_ra_cosdec, pm_dec, radial_velocity) in (mas / yr, mas / yr, km / s)
        [(-0.28301849, -0.12656972, 6.26099034)]>

The underlying representation is still interpolated, and the interpolation
is even kept when transforming frames.

    >>> frame = frame.transform_to(coord.Galactic())
    >>> frame.data[:4]
    <InterpolatedCartesianRepresentation (affine| x, y, z) in Myr| kpc
        [(0.        , -1.8411072 , 1.04913465, 0.71389129),
         (0.25641026, -1.87731612, 1.06955162, 0.69825645),
         (0.51282051, -1.91352503, 1.08996859, 0.68262162),
         (0.76923077, -1.94973395, 1.11038556, 0.66698678)]
     (has differentials w.r.t.: 's')>

For deeper integration and access to interpolated methods, the
``InterpolatedCoordinateFrame`` can wrap any |Frame|, whether
or not it contains an interpolated representation.

    >>> iframe = icoord.InterpolatedCoordinateFrame(frame)  # frame contains irep
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

    >>> frame = coord.ICRS(rep)  # no interpolation (e.g. irep)
    >>> iframe = icoord.InterpolatedCoordinateFrame(frame, affine=affine)
    >>> iframe[:2]
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


    >>> iframe.derivative()[:4]
    <InterpolatedCartesianDifferential (affine| d_x, d_y, d_z) in Myr| kpc / Myr
        [(0.        , 0.1, 0.1, 0.1), (0.25641026, 0.1, 0.1, 0.1),
         (0.51282051, 0.1, 0.1, 0.1), (0.76923077, 0.1, 0.1, 0.1)]>


SkyCoord
--------

There are also interpolated |SkyCoord|. This is actually a direct subclass
of SkyCoord, not a proxy class like the interpolated representations and
coordinate frame. As such, ``InterpolatedSkyCoord`` can be instantiated in
all the normal ways, except that it requires the kwarg ``affine``.

    >>> isc = icoord.InterpolatedSkyCoord(
    ...         [1, 2, 3, 4], [-30, 45, 8, 16],
    ...         frame="icrs", unit="deg",
    ...         affine=affine[:4])
    >>> isc
    <InterpolatedSkyCoord (ICRS): (affine| ra, dec) in Myr| deg
        [(0.        , 1., -30.), (0.25641026, 2.,  45.),
         (0.51282051, 3.,   8.), (0.76923077, 4.,  16.)]>


The only case when |SkyCoord| doesn't need ``affine`` is if it is wrapping an interpolated |Frame|.

    >>> isc = icoord.InterpolatedSkyCoord(iframe)
    >>> isc[:4]
    <InterpolatedSkyCoord (ICRS): (ra, dec, distance) in (deg, deg, kpc)
        [(90.        , 63.43494882, 2.23606798),
         (88.56790382, 63.13836438, 2.27064276),
         (87.20729763, 62.83721465, 2.30555457),
         (85.91438322, 62.53280357, 2.34078832)]
     (pm_ra_cosdec, pm_dec, radial_velocity) in (mas / yr, mas / yr, km / s)
        [(-0.28301849, -0.12656972, 6.26099034),
         (-0.2716564 , -0.12884151, 6.33590983),
         (-0.26078887, -0.13051534, 6.40935614),
         (-0.25040783, -0.13166259, 6.48140523)]>


Like for |Frame|, ``InterpolatedSkyCoord`` preserves the interpolation when transformed between frames and representations.

    >>> isc.transform_to("galactocentric")[:4]
    <InterpolatedSkyCoord (Galactocentric: galcen_coord=<ICRS Coordinate: (ra, dec) in deg
    (266.4051, -28.936175)>, galcen_distance=8.122 kpc, galcen_v_sun=(12.9, 245.6, 7.78) km / s, z_sun=20.8 pc, roll=0.0 deg): (x, y, z) in kpc
        [( -9.96124634, 1.04913531, 0.73940283),
         ( -9.99749514, 1.06955234, 0.72386075),
         (-10.03374393, 1.08996937, 0.70831867),
         (-10.06999273, 1.1103864 , 0.69277659)]
     (v_x, v_y, v_z) in km / s
        [(6.81961773, 249.03792764, 6.68017958),
         (6.78336893, 249.05834467, 6.6646375 ),
         (6.74712013, 249.0787617 , 6.64909542),
         (6.71087133, 249.09917872, 6.63355334)]>


Interpolation means ``InterpolatedSkyCoord`` can be evaluated anywhere between the affine parameter endpoints

    >>> isc(4.8 * u.Gyr)
    <SkyCoord (ICRS): (ra, dec, distance) in (deg, deg, kpc)
        (45.05956281, 35.34846733, 833.11537997)
     (pm_ra_cosdec, pm_dec, radial_velocity) in (mas / yr, mas / yr, km / s)
        (-1.11640239e-06, -5.17923565e-07, 838.31378839)>


``InterpolatedSkyCoord`` can also be differentiated.

    >>> isc.derivative()[:4]
    <InterpolatedCartesianDifferential (affine| d_x, d_y, d_z) in Myr| kpc / Myr
        [(0.        , 0.1, 0.1, 0.1), (0.25641026, 0.1, 0.1, 0.1),
         (0.51282051, 0.1, 0.1, 0.1), (0.76923077, 0.1, 0.1, 0.1)]>



Splines, with units
-------------------

`scipy splines <https://docs.scipy.org/doc/scipy/reference/interpolate.html>`_ do not support |astropy| quantities with units.
The standard workaround solution is to strip the quantities of their units,
apply the interpolation, then add the units back.

As an example:

    >>> import numpy as np, astropy.units as u
    >>> from scipy.interpolate import InterpolatedUnivariateSpline
    >>> x = np.linspace(-3, 3, 50) * u.s
    >>> y = 8 * u.m / (x.value**2 + 4)

    >>> spl = InterpolatedUnivariateSpline(x.to_value(u.s), y.to_value(u.m))
    >>> xs = np.linspace(-2, 2, 10) * u.s  # For evaluating the spline
    >>> y_ntrp = spl(xs.to_value(u.s)) * u.m  # Evaluate, adding back units
    >>> y_ntrp
    <Quantity [1.00000009, 1.24615404, 1.52830261, 1.79999996, 1.97560874,
               1.97560874, 1.79999996, 1.52830261, 1.24615404, 1.00000009] m>


This is fine, but a bit of a hassle. Instead, we can wrap the unit stripping /
adding process into a unit-aware version of the spline interpolation classes.

The same example as above, but with the new class:

    >>> from interpolated_coordinates.utils import InterpolatedUnivariateSplinewithUnits
    >>> spl = InterpolatedUnivariateSplinewithUnits(x, y)
    >>> spl(xs)
    <Quantity [1.00000009, 1.24615404, 1.52830261, 1.79999996, 1.97560874,
               1.97560874, 1.79999996, 1.52830261, 1.24615404, 1.00000009] m>


These splines underpin the interpolated coordinates, above.


References
----------
.. [Dierckx] Paul Dierckx, Curve and Surface Fitting with Splines,
    Oxford University Press, 1993
.. [scipy] Virtanen, P., Gommers, R., Oliphant, M., Reddy, T., Cournapeau,
    E., Peterson, P., Weckesser, J., Walt, M., Wilson, J., Millman, N., Nelson,
    A., Jones, R., Larson, E., Carey, ., Feng, Y., Moore, J., Laxalde, D.,
    Perktold, R., Henriksen, I., Quintero, C., Archibald, A., Pedregosa, P.,
    & SciPy 1.0 Contributors (2020). SciPy 1.0: Fundamental Algorithms for
    Scientific Computing in Python. Nature Methods, 17, 261–272.
