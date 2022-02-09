# -*- coding: utf-8 -*-

"""
A module for scipy splines classes with :mod:`~astropy.units` support.
`scipy` [scipy]_, [Dierckx]_ splines do not support |Quantities| because
they do not understand |Unit|. A standard workaround solution when one needs
to interpolate is to strip quantities of their units, apply the interpolation,
then add units back.

As an example:

.. code-block:: python

    >>> import numpy as np
    >>> import astropy.units as u
    >>> x = np.linspace(-3, 3, 50) * u.s
    >>> y = 8 * u.m / (x.value**2 + 4)
    >>> xs = np.linspace(-2, 2, 10) * u.s  # for evaluating spline

.. code-block:: python

    >>> from scipy.interpolate import InterpolatedUnivariateSpline
    >>> spl = InterpolatedUnivariateSpline(x.to_value(u.s), y.to_value(u.m))
    >>> spl(xs.to_value(u.s)) * u.m  # evaluate, adding back units
    <Quantity [1.00000009, 1.24615404, 1.52830261, 1.79999996, 1.97560874,
               1.97560874, 1.79999996, 1.52830261, 1.24615404, 1.00000009] m>


This is fine, but a bit of a hassle. Instead, we can wrap the unit stripping /
adding process into a unit-aware version of the spline interpolation classes.

The same example as above, but with the new class:

.. code-block:: python
    :emphasize-lines: 1

    >>> from interpolated_coordinates.utils import InterpolatedUnivariateSplinewithUnits
    >>> spl = InterpolatedUnivariateSplinewithUnits(x, y)
    >>> spl(xs)
    <Quantity [1.00000009, 1.24615404, 1.52830261, 1.79999996, 1.97560874,
               1.97560874, 1.79999996, 1.52830261, 1.24615404, 1.00000009] m>

Using :class:`~interpolated_coordinates.utils.InterpolatedUnivariateSplinewithUnits`,
interpolation with `~numpy.ndarray` AND |Quantities| inputs just work.

Plotting this example:

.. plot::
   :context: close-figs
   :alt: example spline plot.

    import numpy as np
    import astropy.units as u
    from astropy.visualization import quantity_support; quantity_support()

    from interpolated_coordinates.utils import InterpolatedUnivariateSplinewithUnits

    x = np.linspace(-3, 3, num=50) * u.s
    y = 8 * u.m / (x.value**2 + 4)
    spl = InterpolatedUnivariateSplinewithUnits(x, y)
    spl(np.linspace(-2, 2, num=10) * u.s)  # Evaluate spline

    xs = np.linspace(-3, 3, num=1000) * u.s  # for sampling

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ax.plot(xs, spl(xs), c="gray", alpha=0.7, lw=3, label="evaluated spline")
    ax.scatter(x, y, c="r", s=25, label="points")

    ax.set_title("Witch of Agnesi")
    ax.set_xlabel(f"x [{ax.get_xlabel()}]")
    ax.set_ylabel(f"y [{ax.get_ylabel()}]")
    plt.legend()


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
"""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import typing as T
import warnings

# THIRD PARTY
import astropy.units as u
import numpy as np
import scipy.interpolate as _interp
from scipy.interpolate import fitpack

try:
    # THIRD PARTY
    from scipy.interpolate._fitpack2 import _curfit_messages
except ModuleNotFoundError:  # scipy < 1.8
    from scipy.interpolate.fitpack2 import _curfit_messages

# LOCAL
from interpolated_coordinates._type_hints import UnitLikeType

__all__ = [
    "UnivariateSplinewithUnits",
    "InterpolatedUnivariateSplinewithUnits",
    "LSQUnivariateSplinewithUnits",
]

##############################################################################
# PARAMETERS

BBoxType = T.List[T.Optional[u.Quantity]]
USwUType = T.TypeVar("USwUType", bound="UnivariateSplinewithUnits")

##############################################################################
# CODE
##############################################################################


class UnivariateSplinewithUnits(_interp.UnivariateSpline):
    """1-D smoothing spline fit to a given set of data points.

    Fits a spline y = spl(x) of degree `k` to the provided `x`, `y` data.  `s`
    specifies the number of knots by specifying a smoothing condition.

    Parameters
    ----------
    x : (N,) Quantity-like or array-like
        1-D array of independent input data. Must be increasing;
        must be strictly increasing if `s` is 0.
    y : (N,) Quantity-like or array-like
        1-D array of dependent input data, of the same length as `x`.
    w : (N,) array_like, optional
        Weights for spline fitting.  Must be positive.  If None (default),
        weights are all equal.
    bbox : (2,) array_like, optional
        2-sequence specifying the boundary of the approximation interval. If
        None (default), ``bbox=[x[0], x[-1]]``.
    k : int, optional
        Degree of the smoothing spline.  Must be 1 <= `k` <= 5.
        Default is `k` = 3, a cubic spline.
    s : float or None, optional
        Positive smoothing factor used to choose the number of knots.  Number
        of knots will be increased until the smoothing condition is satisfied::

            sum((w[i] * (y[i]-spl(x[i])))**2, axis=0) <= s

        If None (default), ``s = len(w)`` which should be a good value if
        ``1/w[i]`` is an estimate of the standard deviation of ``y[i]``.
        If 0, spline will interpolate through all data points.
    ext : int or str, optional
        Controls the extrapolation mode for elements
        not in the interval defined by the knot sequence.

        * if ext=0 or 'extrapolate', return the extrapolated value.
        * if ext=1 or 'zeros', return 0
        * if ext=2 or 'raise', raise a ValueError
        * if ext=3 of 'const', return the boundary value.

        The default value is 0.

    check_finite : bool, optional
        Whether to check that the input arrays contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination or non-sensical results) if the inputs
        do contain infinities or NaNs.
        Default is False.

    x_unit, y_unit : unit-like or None, optional keyword-only
        The |Unit| of ``x``/``y`` (if not `None`), and to which ``x``/``y``
        will be converted before the value is used in the underlying
        interpolation machinery. If ``x``/``y`` does not have units
        (e.g. is an `~numpy.ndarray`) this cannot not be `None`.

    See Also
    --------
    BivariateSpline :
        a base class for bivariate splines.
    SmoothBivariateSpline :
        a smoothing bivariate spline through the given points
    LSQBivariateSpline :
        a bivariate spline using weighted least-squares fitting
    RectSphereBivariateSpline :
        a bivariate spline over a rectangular mesh on a sphere
    SmoothSphereBivariateSpline :
        a smoothing bivariate spline in spherical coordinates
    LSQSphereBivariateSpline :
        a bivariate spline in spherical coordinates using weighted
        least-squares fitting
    RectBivariateSpline :
        a bivariate spline over a rectangular mesh
    InterpolatedUnivariateSpline :
        a interpolating univariate spline for a given set of data points.
    bisplrep :
        a function to find a bivariate B-spline representation of a surface
    bisplev :
        a function to evaluate a bivariate B-spline and its derivatives
    splrep :
        a function to find the B-spline representation of a 1-D curve
    splev :
        a function to evaluate a B-spline or its derivatives
    sproot :
        a function to find the roots of a cubic B-spline
    splint :
        a function to evaluate the definite integral of a B-spline between two
        given points
    spalde :
        a function to evaluate all derivatives of a B-spline

    Notes
    -----
    The number of data points must be larger than the spline degree `k`.

    **NaN handling**: If the input arrays contain ``nan`` values, the result
    is not useful, since the underlying spline fitting routines cannot deal
    with ``nan``. A workaround is to use zero weights for not-a-number
    data points:

    >>> x, y = np.array([1, 2, 3, 4]) * u.m, np.array([1, np.nan, 3, 4]) * u.s
    >>> w = np.isnan(y)
    >>> y[w] = 0.
    >>> spl = UnivariateSplinewithUnits(x, y, w=~w)

    Notice the need to replace a ``nan`` by a numerical value (precise value
    does not matter as long as the corresponding weight is zero.)

    Examples
    --------
    .. plot::
       :context: close-figs

        import astropy.units as u
        import numpy as np
        import matplotlib.pyplot as plt
        from astropy.visualization import quantity_support; quantity_support()

        x = np.linspace(-3, 3, 50) * u.m
        y = np.exp(-x.value**2) + 0.1 * np.random.randn(50) << u.s
        plt.plot(x, y, 'ro', ms=5)

    Use the default value for the smoothing parameter:

    .. plot::
       :context: close-figs

        import astropy.units as u
        import numpy as np
        import matplotlib.pyplot as plt
        from astropy.visualization import quantity_support; quantity_support()

        from interpolated_coordinates.utils import UnivariateSplinewithUnits

        x = np.linspace(-3, 3, 50) * u.m
        y = np.exp(-x.value**2) + 0.1 * np.random.randn(50) << u.s
        spl = UnivariateSplinewithUnits(x, y)
        xs = np.linspace(-3, 3, 1000) * u.m
        plt.plot(xs, spl(xs), 'g', lw=3)

    Manually change the amount of smoothing:

    .. plot::
       :context: close-figs

        import astropy.units as u
        import numpy as np
        import matplotlib.pyplot as plt
        from astropy.visualization import quantity_support; quantity_support()

        from interpolated_coordinates.utils import UnivariateSplinewithUnits

        x = np.linspace(-3, 3, 50) * u.m
        y = np.exp(-x.value**2) + 0.1 * np.random.randn(50) << u.s
        spl = UnivariateSplinewithUnits(x, y)

        spl.set_smoothing_factor(0.5)
        plt.plot(xs, spl(xs), 'b', lw=3)

    """

    def __init__(
        self,
        x: u.Quantity,
        y: u.Quantity,
        w: T.Optional[np.ndarray] = None,
        bbox: BBoxType = [None, None],
        k: int = 3,
        s: T.Optional[float] = None,
        ext: T.Union[int, str, None] = 0,
        check_finite: bool = False,
        *,
        x_unit: T.Optional[UnitLikeType] = None,
        y_unit: T.Optional[UnitLikeType] = None,
    ):
        # The unit for x and y, respectively. If None (default), gets
        # the units from x and y.
        self._xunit = u.Unit(x_unit) if x_unit is not None else x.unit
        self._yunit = u.Unit(y_unit) if y_unit is not None else y.unit

        # Make x, y to value, so can create IUS as normal
        x = (x << self._xunit).value
        y = (y << self._yunit).value

        if bbox[0] is not None:
            bbox[0] = bbox[0].to(self._xunit).value
        if bbox[1] is not None:
            bbox[1] = bbox[1].to(self._xunit).value

        # Make spline
        super().__init__(x, y, w=w, bbox=bbox, k=k, s=s, ext=ext, check_finite=check_finite)

    @property
    def x_unit(self):
        """|Unit| of the independent data."""
        return self._xunit

    @property
    def y_unit(self):
        """|Unit| of the dependent data."""
        return self._yunit

    def validate_input(
        self,
        x: T.Union[np.ndarray, u.Quantity],
        y: T.Union[np.ndarray, u.Quantity],
        w: T.Union[np.ndarray, u.Quantity],
        bbox: BBoxType,
        k: int,
        s: float,
        ext: T.Union[int, str],
        check_finite: float,
    ) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray, T.List[u.Quantity], T.Union[int, str]]:
        # first validate units
        x = x.to_value(self._xunit) if isinstance(x, u.Quantity) else x
        y = y.to_value(self._yunit) if isinstance(y, u.Quantity) else y

        # then validate with UnivariateSpline method, which works with units!
        out: T.Tuple[np.ndarray, np.ndarray, np.ndarray, T.List[u.Quantity], T.Union[int, str]]
        out = super().validate_input(x, y, w, bbox, k, s, ext, check_finite)
        return out

    @classmethod
    def _from_tck(
        cls,
        tck: T.Tuple[np.ndarray, np.ndarray, np.ndarray],
        x_unit: UnitLikeType,
        y_unit: UnitLikeType,
        ext: int = 0,
    ) -> USwUType:
        """Construct a spline object from given tck."""
        self: USwUType = super()._from_tck(tck, ext=ext)
        self._xunit = u.Unit(x_unit)
        self._yunit = u.Unit(y_unit)

        return self

    def _reset_class(self) -> None:
        data = self._data
        n, t, c, k, ier = data[7], data[8], data[9], data[5], data[-1]
        self._eval_args = t[:n], c[:n], k
        if ier == 0:
            # the spline returned has a residual sum of squares fp
            # such that abs(fp-s)/s <= tol with tol a relative
            # tolerance set to 0.001 by the program
            pass
        elif ier == -1:
            # the spline returned is an interpolating spline
            self._set_class(InterpolatedUnivariateSplinewithUnits)
        elif ier == -2:
            # the spline returned is the weighted least-squares
            # polynomial of degree k. In this extreme case fp gives
            # the upper bound fp0 for the smoothing factor s.
            self._set_class(LSQUnivariateSplinewithUnits)
        else:
            # error
            if ier == 1:
                self._set_class(LSQUnivariateSplinewithUnits)
            message = _curfit_messages.get(ier, "ier=%s" % (ier))
            warnings.warn(message)

    def _set_class(self, cls: type) -> None:
        self._spline_class = cls
        if self.__class__ in (
            UnivariateSplinewithUnits,
            InterpolatedUnivariateSplinewithUnits,
            LSQUnivariateSplinewithUnits,
        ):
            self.__class__ = cls
        else:
            # It's an unknown subclass -- don't change class. cf. #731
            pass

    def __call__(self, x: np.ndarray, nu: int = 0, ext: T.Optional[int] = None) -> u.Quantity:
        """Evaluate spline (or its nu-th derivative) at positions x.

        Parameters
        ----------
        x : ndarray or Quantity array-like
            A 1-D array of points at which to return the value of the smoothed
            spline or its derivatives. Note: `x` can be unordered but the
            evaluation is more efficient if `x` is (partially) ordered.
        nu  : int, optional
            The order of derivative of the spline to compute.
        ext : int, optional
            Controls the value returned for elements of `x` not in the
            interval defined by the knot sequence.

            * if ext=0 or 'extrapolate', return the extrapolated value.
            * if ext=1 or 'zeros', return 0
            * if ext=2 or 'raise', raise a ValueError
            * if ext=3 or 'const', return the boundary value.

            The default value is 0, passed from the initialization of
            UnivariateSpline.

        Returns
        -------
        y : Quantity array_like
            Evaluated spline with units ``y_unit``. Same shape as `x`.
        """
        x = (x << self._xunit).value
        y: np.ndarray = super().__call__(x, nu=nu, ext=ext)
        yq: u.Quantity = y << self._yunit
        return yq

    def get_knots(self) -> u.Quantity:
        """Return positions of interior knots of the spline.

        Internally, the knot vector contains ``2*k`` additional boundary knots.
        Has units of `x` position
        """
        return super().get_knots() * self._xunit

    def get_coeffs(self) -> u.Quantity:
        """Return spline coefficients."""
        return super().get_coeffs() << self._yunit

    def get_residual(self) -> u.Quantity:
        """Return weighted sum of squared residuals of spline approximation.

        This is equivalent to::
            sum((w[i] * (y[i]-spl(x[i])))**2, axis=0)
        """
        return super().get_residual() << self._yunit

    def integral(self, a: u.Quantity, b: u.Quantity) -> u.Quantity:
        r"""Return definite integral of the spline between two given points.

        Parameters
        ----------
        a : Quantity
            Lower limit of integration.
        b : Quantity
            Upper limit of integration.

        Returns
        -------
        integral : float
            The value of the definite integral of the spline between limits.

        Examples
        --------
        >>> from scipy.interpolate import UnivariateSpline
        >>> x = np.linspace(0, 3, 11)
        >>> y = x**2
        >>> spl = UnivariateSpline(x, y)
        >>> spl.integral(0, 3)
        9.0

        which agrees with :math:`\int x^2 dx = x^3 / 3` between the limits
        of 0 and 3.

        A caveat is that this routine assumes the spline to be zero outside of
        the data limits:

        >>> spl.integral(-1, 4)
        9.0

        >>> spl.integral(-1, 0)
        0.0
        """
        a_val: float = a.to_value(self._xunit)
        b_val: float = b.to_value(self._xunit)
        v: float = super().integral(a_val, b_val)
        q: u.Quantity = v << (self._xunit * self._yunit)
        return q

    def derivatives(self, x: u.Quantity) -> np.ndarray:
        """Return all derivatives of the spline at the point x.

        Parameters
        ----------
        x : |Quantity|
            The point to evaluate the derivatives at.

        Returns
        -------
        der : ndarray of Quantity object, shape(k+1,)
            Derivatives of the orders 0 to k.

        Examples
        --------
        >>> from scipy.interpolate import UnivariateSpline
        >>> x = np.linspace(0, 3, 11)
        >>> y = x**2
        >>> spl = UnivariateSpline(x, y)
        >>> spl.derivatives(1.5)  # doctest: +FLOAT_CMP
        array([2.25, 3.  , 2.  , 0.  ])
        """
        x_val: float = x.to_value(self._xunit)
        d_vals: np.ndarray = super().derivatives(x_val)
        return np.array(
            [d * self._yunit / self._xunit ** i for i, d in enumerate(d_vals)],
            dtype=u.Quantity,
        )

    def roots(self) -> u.Quantity:
        """Return the zeros of the spline.

        Restriction: only cubic splines are supported by fitpack.
        """
        return super().roots() * self._xunit

    def derivative(self, n: int = 1) -> USwUType:
        r"""Construct a new spline representing the derivative of this spline.

        Parameters
        ----------
        n : int, optional
            Order of derivative to evaluate. Default: 1

        Returns
        -------
        spline : UnivariateSpline
            Spline of order k2=k-n representing the derivative of this
            spline.

        See Also
        --------
        splder, antiderivative

        Examples
        --------
        This can be used for finding maxima of a curve:

        >>> from interpolated_coordinates.utils import UnivariateSplinewithUnits
        >>> x = np.linspace(0, 10, 70) * u.s
        >>> y = np.sin(x.value) * u.m
        >>> spl = UnivariateSplinewithUnits(x, y, k=4, s=0)

        Now, differentiate the spline and find the zeros of the
        derivative. (NB: `sproot` only works for order 3 splines, so we
        fit an order 4 spline):

        >>> spl.derivative().roots() / np.pi  # doctest: +FLOAT_CMP
        <Quantity [0.50000001, 1.5       , 2.49999998] s>

        This agrees well with roots :math:`\\pi/2 + n\\pi` of
        :math:`\\cos(x) = \\sin'(x)`.
        """
        tck = fitpack.splder(self._eval_args, n)
        # if self.ext is 'const', derivative.ext will be 'zeros'
        ext = 1 if self.ext == 3 else self.ext
        x_unit = self._xunit
        y_unit = self._yunit / self._xunit ** n
        return self.__class__._from_tck(tck, x_unit=x_unit, y_unit=y_unit, ext=ext)

    def antiderivative(self, n: int = 1) -> USwUType:
        r"""Construct a new spline representing this spline's antiderivative.

        Parameters
        ----------
        n : int, optional
            Order of antiderivative to evaluate. Default: 1

        Returns
        -------
        spline : `~interpolated_coordinates.utils.UnivariateSplinewithUnits`
            Spline of order k2=k+n representing the antiderivative of this
            spline.

        See Also
        --------
        splantider, derivative

        Examples
        --------
        >>> from scipy.interpolate import UnivariateSpline
        >>> x = np.linspace(0, np.pi/2, 70)
        >>> y = 1 / np.sqrt(1 - 0.8*np.sin(x)**2)
        >>> spl = UnivariateSpline(x, y, s=0)

        The derivative is the inverse operation of the antiderivative,
        although some floating point error accumulates:

        >>> spl(1.7) - spl.antiderivative().derivative()(1.7) != 0
        True

        Antiderivative can be used to evaluate definite integrals:

        >>> ispl = spl.antiderivative()
        >>> ispl(np.pi/2) - ispl(0)  # doctest: +FLOAT_CMP
        2.2572053588768486

        This is indeed an approximation to the complete elliptic integral
        :math:`K(m) = \\int_0^{\\pi/2} [1 - m\\sin^2 x]^{-1/2} dx`:

        >>> from scipy.special import ellipk
        >>> ellipk(0.8)  # doctest: +FLOAT_CMP
        2.2572053268208538
        """
        tck = fitpack.splantider(self._eval_args, n)
        x_unit = self._xunit
        y_unit = self._yunit * self._xunit ** n
        return self.__class__._from_tck(tck, x_unit=x_unit, y_unit=y_unit, ext=self.ext)


# -------------------------------------------------------------------


class InterpolatedUnivariateSplinewithUnits(
    UnivariateSplinewithUnits,
    _interp.InterpolatedUnivariateSpline,
):
    """1-D interpolating spline for a given set of data points, with units.

    Fits a spline y = spl(x) of degree `k` to the provided `x`, `y` data.
    Spline function passes through all provided points. Equivalent to
    `UnivariateSpline` with s=0.

    Parameters
    ----------
    x : (N,) |Quantity| array_like
        Input dimension of data points -- must be strictly increasing
    y : (N,) |Quantity| array_like
        input dimension of data points
    w : (N,) |Quantity| array_like, optional
        Weights for spline fitting.  Must be positive.  If None (default),
        weights are all equal.
    bbox : (2,) |Quantity| array_like, optional
        2-sequence specifying the boundary of the approximation interval. If
        None (default), ``bbox=[x[0], x[-1]]``.
    k : int, optional
        Degree of the smoothing spline.  Must be 1 <= `k` <= 5.
    ext : int or str, optional
        Controls the extrapolation mode for elements
        not in the interval defined by the knot sequence.

        * if ext=0 or 'extrapolate', return the extrapolated value.
        * if ext=1 or 'zeros', return 0
        * if ext=2 or 'raise', raise a ValueError
        * if ext=3 of 'const', return the boundary value.

        The default value is 0.

    check_finite : bool, optional
        Whether to check that the input arrays contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination or non-sensical results) if the inputs
        do contain infinities or NaNs.
        Default is False.

    x_unit, y_unit : unit-like or None, optional keyword-only
        The |Unit| of ``x``/``y`` (if not `None`), and to which ``x``/``y``
        will be converted before the value is used in the underlying
        interpolation machinery. If ``x``/``y`` does not have units
        (e.g. is an `~numpy.ndarray`) this cannot not be `None`.

    See Also
    --------
    UnivariateSpline : Superclass -- allows knots to be selected by a
        smoothing condition
    LSQUnivariateSpline : spline for which knots are user-selected
    splrep : An older, non object-oriented wrapping of FITPACK
    splev, sproot, splint, spalde
    BivariateSpline : A similar class for two-dimensional spline interpolation

    Notes
    -----
    The number of data points must be larger than the spline degree `k`.

    Examples
    --------
    >>> import numpy as np
    >>> import astropy.units as u
    >>> from interpolated_coordinates.utils import InterpolatedUnivariateSplinewithUnits
    >>> x = np.linspace(-3, 3, 50) * u.s
    >>> y = 8 * u.m / (x.value**2 + 4)
    >>> spl = InterpolatedUnivariateSplinewithUnits(x, y)

    .. plot::
        :context: close-figs

        import astropy.units as u
        import numpy as np
        import matplotlib.pyplot as plt
        from astropy.visualization import quantity_support; quantity_support()

        from interpolated_coordinates.utils import InterpolatedUnivariateSplinewithUnits

        x = np.linspace(-3, 3, 50) * u.s
        y = 8 * u.m / (x.value**2 + 4)
        spl = InterpolatedUnivariateSplinewithUnits(x, y)

        plt.plot(x, y, 'ro', ms=5)
        xs = np.linspace(-3, 3, 1000) * u.s
        plt.plot(xs, spl(xs), 'g', lw=3, alpha=0.7)

    Notice that the ``spl(x)`` interpolates `y`:

    >>> spl.get_residual()
    <Quantity 0. m>
    """

    def __init__(
        self,
        x: u.Quantity,
        y: u.Quantity,
        w: T.Optional[np.ndarray] = None,
        bbox: BBoxType = [None, None],
        k: int = 3,
        ext: int = 0,
        check_finite: bool = False,
        *,
        x_unit: T.Optional[UnitLikeType] = None,
        y_unit: T.Optional[UnitLikeType] = None,
    ):
        # The unit for x and y, respectively. If None (default), gets
        # the units from x and y.
        self._xunit = u.Unit(x_unit) if x_unit is not None else x.unit
        self._yunit = u.Unit(y_unit) if y_unit is not None else y.unit

        # Make x, y to value, so can create IUS as normal
        x = (x << self._xunit).value
        y = (y << self._yunit).value

        if bbox[0] is not None:
            bbox[0] = bbox[0].to(self._xunit).value
        if bbox[1] is not None:
            bbox[1] = bbox[1].to(self._xunit).value

        # Make spline
        _interp.InterpolatedUnivariateSpline.__init__(
            self,
            x,
            y,
            w=w,
            bbox=bbox,
            k=k,
            ext=ext,
            check_finite=check_finite,
        )


# -------------------------------------------------------------------


class LSQUnivariateSplinewithUnits(UnivariateSplinewithUnits, _interp.LSQUnivariateSpline):
    """1-D spline with explicit internal knots.

    Fits a spline y = spl(x) of degree `k` to the provided `x`, `y` data.  `t`
    specifies the internal knots of the spline

    Parameters
    ----------
    x : (N,) array_like
        Input dimension of data points -- must be increasing
    y : (N,) array_like
        Input dimension of data points
    t : (M,) array_like
        interior knots of the spline.  Must be in ascending order and::

            bbox[0] < t[0] < ... < t[-1] < bbox[-1]

    w : (N,) array_like, optional
        weights for spline fitting. Must be positive. If None (default),
        weights are all equal.
    bbox : (2,) array_like, optional
        2-sequence specifying the boundary of the approximation interval. If
        None (default), ``bbox = [x[0], x[-1]]``.
    k : int, optional
        Degree of the smoothing spline.  Must be 1 <= `k` <= 5.
        Default is `k` = 3, a cubic spline.
    ext : int or str, optional
        Controls the extrapolation mode for elements
        not in the interval defined by the knot sequence.

        * if ext=0 or 'extrapolate', return the extrapolated value.
        * if ext=1 or 'zeros', return 0
        * if ext=2 or 'raise', raise a ValueError
        * if ext=3 of 'const', return the boundary value.

        The default value is 0.

    check_finite : bool, optional
        Whether to check that the input arrays contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination or non-sensical results) if the inputs
        do contain infinities or NaNs.
        Default is False.

    x_unit, y_unit : unit-like or None, optional keyword-only
        The |Unit| of ``x``/``y`` (if not `None`), and to which ``x``/``y``
        will be converted before the value is used in the underlying
        interpolation machinery. If ``x``/``y`` does not have units
        (e.g. is an `~numpy.ndarray`) this cannot not be `None`.

    Raises
    ------
    ValueError
        If the interior knots do not satisfy the Schoenberg-Whitney conditions.

    See Also
    --------
    UnivariateSpline :
        a smooth univariate spline to fit a given set of data points.
    InterpolatedUnivariateSpline :
        a interpolating univariate spline for a given set of data points.
    splrep :
        a function to find the B-spline representation of a 1-D curve
    splev :
        a function to evaluate a B-spline or its derivatives
    sproot :
        a function to find the roots of a cubic B-spline
    splint :
        a function to evaluate the definite integral of a B-spline between two
        given points
    spalde :
        a function to evaluate all derivatives of a B-spline

    Notes
    -----
    The number of data points must be larger than the spline degree `k`.

    Knots `t` must satisfy the Schoenberg-Whitney conditions,
    i.e., there must be a subset of data points ``x[j]`` such that
    ``t[j] < x[j] < t[j+k+1]``, for ``j=0, 1,...,n-k-2``.

    Examples
    --------
    >>> import numpy as np
    >>> import astropy.units as u
    >>> import matplotlib.pyplot as plt
    >>> from interpolated_coordinates.utils import LSQUnivariateSplinewithUnits

    >>> x = np.linspace(-3, 3, 50) * u.m
    >>> y = np.exp(-x.value**2) + 0.1 * np.random.randn(50) << u.s

    Fit a smoothing spline with a pre-defined internal knots:

    >>> t = [-1, 0, 1]
    >>> spl = LSQUnivariateSplinewithUnits(x, y, t)

    .. plot::
       :context: close-figs

        import astropy.units as u
        import numpy as np
        import matplotlib.pyplot as plt
        from astropy.visualization import quantity_support; quantity_support()

        from interpolated_coordinates.utils import LSQUnivariateSplinewithUnits

        x = np.linspace(-3, 3, 50) * u.m
        y = np.exp(-x.value**2) + 0.1 * np.random.randn(50) << u.s

        t = [-1, 0, 1]
        spl = LSQUnivariateSplinewithUnits(x, y, t)

        xs = np.linspace(-3, 3, 1000) * u.m
        plt.plot(x, y, 'ro', ms=5)
        plt.plot(xs, spl(xs), 'g-', lw=3)

    Check the knot vector:

    >>> spl.get_knots()
    <Quantity [-3., -1.,  0.,  1.,  3.] m>

    Constructing lsq spline using the knots from another spline:

    >>> x = np.arange(10) * u.m
    >>> s = UnivariateSplinewithUnits(x, x, s=0)
    >>> s.get_knots()
    <Quantity [0., 2., 3., 4., 5., 6., 7., 9.] m>

    >>> knt = s.get_knots()
    >>> s1 = LSQUnivariateSplinewithUnits(x, x, knt[1:-1])  # Chop 1st and last knot
    >>> s1.get_knots()
    <Quantity [0., 2., 3., 4., 5., 6., 7., 9.] m>
    """

    def __init__(
        self,
        x: u.Quantity,
        y: u.Quantity,
        t: u.Quantity,
        w: T.Optional[np.ndarray] = None,
        bbox: BBoxType = [None, None],
        k: int = 3,
        ext: int = 0,
        check_finite: bool = False,
        *,
        x_unit: T.Optional[UnitLikeType] = None,
        y_unit: T.Optional[UnitLikeType] = None,
    ) -> None:
        # The unit for x and y, respectively. If None (default), gets
        # the units from x and y.
        self._xunit = u.Unit(x_unit) if x_unit is not None else x.unit
        self._yunit = u.Unit(y_unit) if y_unit is not None else y.unit

        # Make x, y, t to value, so can create IUS as normal
        x = (x << self._xunit).value
        y = (y << self._yunit).value
        t = (t << self._xunit).value

        if bbox[0] is not None:
            bbox[0] = bbox[0].to(self._xunit).value
        if bbox[1] is not None:
            bbox[1] = bbox[1].to(self._xunit).value

        _interp.LSQUnivariateSpline.__init__(
            self,
            x,
            y,
            t,
            w=w,
            bbox=bbox,
            k=k,
            ext=ext,
            check_finite=check_finite,
        )
