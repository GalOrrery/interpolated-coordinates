# -*- coding: utf-8 -*-

from __future__ import annotations

# STDLIB
import abc
import copy
import inspect
import typing as T

# THIRD PARTY
import astropy.coordinates as coord
import astropy.coordinates.representation as r
import astropy.units as u
import numpy as np
import numpy.lib.recfunctions as rfn
from astropy.coordinates.representation import _array2string
from numpy import array_equal

# LOCAL
from . import _type_hints as TH
from .utils import GenericDifferential
from .utils import InterpolatedUnivariateSplinewithUnits as IUSU

__all__ = [
    "InterpolatedBaseRepresentationOrDifferential",
    "InterpolatedRepresentation",
    "InterpolatedCartesianRepresentation",
    "InterpolatedDifferential",
]

##############################################################################
# PARAMETERS

_UNIT_DIF_TYPES = (  # the unit-differentials
    r.UnitSphericalDifferential,
    r.UnitSphericalCosLatDifferential,
    r.RadialDifferential,
)

IRoDType = T.TypeVar("IRoDType", bound="InterpolatedBaseRepresentationOrDifferential")
IRType = T.TypeVar("IRType", bound="InterpolatedRepresentation")
ICRType = T.TypeVar("ICRType", bound="InterpolatedCartesianRepresentation")
IDType = T.TypeVar("IDType", bound="InterpolatedDifferential")
DType = T.TypeVar("DType", bound=r.BaseDifferential)


_op_msg = "can only {} {} if the interpolation variables are the same."


##############################################################################
# CODE
##############################################################################


def _find_first_best_compatible_differential(
    rep: r.BaseRepresentation,
    n: int = 1,
) -> T.Union[r.BaseDifferential, GenericDifferential]:
    """Find a compatible differential.

    There can be more than one, so we select the first one.

    """
    # get names of derivatives wrt the affine parameter
    pkeys = {"d_" + k for k in rep.components}

    # then get compatible differential classes (by matching keys)
    dif_comps = [
        cls
        for cls in rep._compatible_differentials  # the options
        if pkeys == set(cls.attr_classes.keys())  # key match
    ]

    if dif_comps:  # not empty. Can't tell them apart, so the first will do
        derivative_type = dif_comps[0]

    # TODO uncomment when encounter (then can also write test)
    # else:  # nothing matches, so we make a differential
    #     derivative_type = _make_generic_differential_for_representation(
    #         rep.__class__,
    #         n=n,
    #     )

    if n != 1:
        derivative_type = GenericDifferential._make_generic_cls(derivative_type, n=n)

    return derivative_type


def _infer_derivative_type(
    rep: r.BaseRepresentationOrDifferential,
    dif_unit: TH.UnitLikeType,
    n: int = 1,
) -> T.Union[r.BaseDifferential, GenericDifferential]:
    """Infer the Differential class used in a derivative wrt time.

    If it can't infer the correct differential class, defaults
    to `~interpolated_coordinates.utils.GenericDifferential`.

    Checks compatible differentials for classes with matching names.

    Parameters
    ----------
    rep : `~astropy.coordinates.BaseRepresentationOrDifferential` instance
        The representation object
    dif_unit : unit-like
        The differential unit
    n : int
        The order of the derivative

    Returns
    -------
    `astropy.coordinates.BaseDifferential` or `~interpolated_coordinates.utils.GenericDifferential`  # noqa: E501
    """
    unit = u.Unit(dif_unit)
    rep_cls = rep.__class__  # (store rep class for line length)

    # Now check we can even do this: if can't make a better Generic
    # 1) can't for `Differentials` and stuff without compatible diffs
    if isinstance(rep, r.BaseDifferential):
        derivative_type = GenericDifferential._make_generic_cls(rep_cls, n=n + 1)
    # 2) can't for non-time derivatives
    elif unit.physical_type != "time":
        derivative_type = GenericDifferential._make_generic_cls_for_representation(rep_cls, n=n)

    else:  # Differentiating a Representation wrt time
        derivative_type = _find_first_best_compatible_differential(rep, n=n)

    return derivative_type


##############################################################################


class InterpolatedBaseRepresentationOrDifferential:
    """Wrapper for Representations, adding affine interpolations.

    .. todo::

        - override all the methods, mapping to underlying Representation
        - figure out how to do ``from_cartesian`` as a class method
        - get ``X_interp`` as properties. Need to do ``__dict__`` manipulation,
          like BaseCoordinateFrame
        - pass through derivative_type in all methods!

    Parameters
    ----------
    rep : `~astropy.coordinates.BaseRepresentation` instance
    affine : `~astropy.units.Quantity` array-like
        The affine interpolation parameter.

    interps : Mapping or None (optional, keyword-only)
        Has same structure as a Representation

        .. code-block:: text

            dict(component name: interpolation,
                ...
                "differentials": dict(
                    "s" : dict(component name: interpolation, ...),
                    ...))

    **interp_kwargs
        Only used if `interps` is None.
        keyword arguments into interpolation class

    Other Parameters
    ----------------
    interp_cls : Callable (optional, keyword-only)
        option for 'interp_kwargs'.
        If not specified, default is `IUSU`.

    derivative_type : Callable (optional, keyword-only)
        The class to use when differentiating wrt to the affine parameter.
        If not provided, will use `_infer_derivative_type` to infer.
        Defaults to `GenericDifferential` if all else fails.

    Raises
    ------
    ValueError
        If `rep` is a BaseRepresentationOrDifferential class, not instance
        If affine shape is not 1-D.
        If affine is not same length as `rep`.
    TypeError
        If `rep` not not type BaseRepresentationOrDifferential.
    """

    def __new__(cls: T.Type[IRoDType], *args: T.Any, **kwargs: T.Any) -> IRoDType:
        if cls is InterpolatedBaseRepresentationOrDifferential:
            raise TypeError(f"Cannot instantiate a {cls}.")

        inst: IRoDType = super().__new__(cls)
        return inst

    def __init__(
        self,
        rep: r.BaseRepresentationOrDifferential,
        affine: u.Quantity,
        *,
        interps: T.Optional[T.Dict] = None,
        derivative_type: T.Optional[r.BaseDifferential] = None,
        **interp_kwargs: T.Any,
    ) -> None:
        # Check it's instantiated and right class
        if inspect.isclass(rep) and issubclass(
            rep,
            r.BaseRepresentationOrDifferential,
        ):
            raise ValueError("Must instantiate `rep`.")
        elif not isinstance(rep, r.BaseRepresentationOrDifferential):
            raise TypeError("`rep` must be a `BaseRepresentationOrDifferential`.")

        # Affine parameter
        affine = u.Quantity(affine, copy=False)  # ensure Quantity
        if not affine.ndim == 1:
            raise ValueError("`affine` must be 1-D.")
        elif len(affine) != len(rep):
            raise ValueError("`affine` must be same length as `rep`")

        # store representation and affine parameter
        self.data = rep
        self._affine = affine = u.Quantity(affine, copy=True)  # TODO copy?

        # The class to use when differentiating wrt to the affine parameter.
        if derivative_type is None:
            derivative_type = _infer_derivative_type(rep, affine.unit)
        # TODO better detection if derivative_type doesn't work!
        self._derivative_type = derivative_type
        self._derivatives: T.Dict[str, T.Any] = dict()

        # -----------------------
        # Construct interpolation

        self._interp_kwargs = interp_kwargs.copy()  # TODO need copy?

        if interps is not None:
            self._interps = interps
        else:
            # determine interpolation type
            interp_cls = interp_kwargs.pop("interp_cls", IUSU)

            self._interps = dict()
            # positional information
            for comp in rep.components:
                self._interps[comp] = interp_cls(affine, getattr(rep, comp), **interp_kwargs)

            # differentials information
            # these are stored in a dictionary with keys wrt time
            # ex : rep.differentials["s"] is a Velocity
            if hasattr(rep, "differentials"):
                for k, differential in rep.differentials.items():
                    d_derivative_type: T.Optional[type]

                    # Is this already an InterpolatedDifferential?
                    # then need to pop back to the Differential
                    if isinstance(differential, InterpolatedDifferential):
                        d_derivative_type = differential.derivative_type
                        differential = differential.data
                    else:
                        d_derivative_type = None

                    # interpolate differential
                    dif = InterpolatedDifferential(
                        differential,
                        affine,
                        interp_cls=interp_cls,
                        derivative_type=d_derivative_type,
                        **interp_kwargs,
                    )

                    # store in place of original
                    self.data.differentials[k] = dif

    @property
    def affine(self) -> u.Quantity:  # read-only
        return self._affine

    @property
    def _class_(self: IRoDType) -> T.Type[IRoDType]:
        """Get this object's true class, not the un-interpolated class."""
        cls: T.Type[IRoDType] = object.__class__(self)
        return cls

    def _realize_class(self: IRoDType, rep: r.BaseRepresentation, affine: u.Quantity) -> IRoDType:
        inst: IRoDType = self._class_(
            rep, affine, derivative_type=self.derivative_type, **self._interp_kwargs
        )
        return inst

    #################################################################
    # Interpolation Methods

    @abc.abstractmethod
    def __call__(self, affine: T.Optional[u.Quantity] = None) -> IRoDType:
        """Evaluate interpolated representation.

        Parameters
        ----------
        affine : `~astropy.units.Quantity` array-like
            The affine interpolation parameter.
            If None, returns representation points.
        """

    @property
    def derivative_type(self) -> type:
        """The class used when taking a derivative."""
        return self._derivative_type

    @derivative_type.setter
    def derivative_type(self, value: type) -> None:
        """The class used when taking a derivative."""
        self._derivative_type = value
        self.clear_derivatives()

    def clear_derivatives(self: IRoDType) -> IRoDType:
        """Return self, clearing cached derivatives."""
        if hasattr(self, "_derivatives"):
            for key in tuple(self._derivatives.keys()):  # iter over fixed keys list
                if key.startswith("affine "):
                    self._derivatives.pop(key)
        return self

    def derivative(self, n: int = 1) -> InterpolatedDifferential:
        r"""Construct a new spline representing the derivative of this spline.

        .. todo::

            Keep the derivatives of the differentials

        Parameters
        ----------
        n : int, optional
            Order of derivative to evaluate. Default: 1
        """
        # Evaluate the spline on each argument of the position
        params = {
            (k if k.startswith("d_") else "d_" + k): interp.derivative(n=n)(self.affine)
            for k, interp in self._interps.items()
        }

        if n == 1:
            deriv_cls = self.derivative_type
        else:
            deriv_cls = _infer_derivative_type(self.data, self.affine.unit, n=n)

        # Make Differential
        deriv = deriv_cls(**params)

        # Interpolate
        ideriv = InterpolatedDifferential(deriv, self.affine, **self._interp_kwargs)
        # TODO! rare case when differentiating an integral of a Representation
        # then want to return an interpolated Representation!

        return ideriv

    # def antiderivative(self, n: int = 1) -> T.Any:
    #     r"""Construct a new spline representing the integral of this spline.
    #
    #     Parameters
    #     ----------
    #     n : int, optional
    #         Order of derivative to evaluate. Default: 1
    #     """
    #     # evaluate the spline on each argument of the position
    #     params = {
    #         k.lstrip("d_"): interp.antiderivative(n=n)(self.affine)
    #         for k, interp in self._interps.items()
    #     }
    #     # deriv = GenericDifferential(*params)
    #     # return self._class_(deriv, self.affine, **self._interp_kwargs)
    #     return params

    # def integral(self, a, b):
    #     """Return definite integral between two given points."""
    #     raise NotImplementedError("What does this even mean?")

    #################################################################
    # Mapping to Underlying Representation

    # ---------------------------------------------------------------
    # Hidden methods

    @property
    def __class__(self) -> T.Type[r.BaseRepresentationOrDifferential]:
        """Make class appear the same as the underlying Representation."""
        cls: T.Type[r.BaseRepresentationOrDifferential] = self.data.__class__
        return cls

    @__class__.setter
    def __class__(self, value: T.Any) -> None:  # needed for mypy  # noqa: F811
        raise TypeError("cannot set attribute ``__class__``.")

    def __getattr__(self, key: str) -> T.Any:
        """Route everything to underlying Representation."""
        got: T.Any = getattr(self.data, key)
        return got

    def __getitem__(self: IRoDType, key: T.Union[str, slice, np.ndarray]) -> IRoDType:
        """Getitem on Representation, re-interpolating."""
        rep: r.BaseRepresentation = self.data[key]
        afn: u.Quantity = self.affine[key]
        inst: IRoDType = self._realize_class(rep, afn)
        return inst

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        """String representation, adding interpolation information."""
        prefixstr = "    "
        values = rfn.merge_arrays((self.affine.value, self.data._values), flatten=True)
        arrstr = _array2string(values, prefix=prefixstr)

        diffstr = ""
        if getattr(self, "differentials", None):
            diffstr = "\n (has differentials w.r.t.: {})".format(
                ", ".join([repr(key) for key in self.differentials.keys()]),
            )

        aurep = str(self.affine.unit) or "[dimensionless]"

        _unitstr = self.data._unitstr
        if _unitstr:
            if _unitstr[0] == "(":
                unitstr = "in " + "(" + aurep + "| " + _unitstr[1:]
            else:
                unitstr = "in " + aurep + "| " + _unitstr
        else:
            unitstr = f"{aurep}| [dimensionless]"

        s: str = "<Interpolated{} (affine| {}) {:s}\n{}{}{}>".format(
            self.__class__.__name__,
            ", ".join(self.data.components),
            unitstr,
            prefixstr,
            arrstr,
            diffstr,
        )
        return s

    def _scale_operation(
        self: IRoDType, op: T.Callable, *args: T.Any, scaled_base: bool = False
    ) -> IRoDType:
        rep = self.data._scale_operation(op, *args, scaled_base=scaled_base)
        inst: IRoDType = self._realize_class(rep, self.affine)
        return inst

    # ---------------------------------------------------------------
    # Math Methods

    def __add__(
        self: IRoDType,
        other: T.Union[r.BaseRepresentationOrDifferential, IRoDType],
    ) -> IRoDType:
        """Add other to an InterpolatedBaseRepresentationOrDifferential

        If other:
        - point : add to data, keep affine the same, re-interpolate
        - vector : add to data, keep affine the same, re-interpolate
        - interpolated : must be same interpolation!
        """
        if isinstance(other, InterpolatedBaseRepresentationOrDifferential):
            if not array_equal(other.affine, self.affine):
                raise ValueError(_op_msg.format("add", self._class_))
        # Add, then re-interpolate
        return self._realize_class(self.data.__add__(other), self.affine)

    def __sub__(
        self: IRoDType,
        other: T.Union[IRoDType, r.BaseRepresentationOrDifferential],
    ) -> IRoDType:
        """Add other to an InterpolatedBaseRepresentationOrDifferential

        If other:
        - point : add to data, keep affine the same, re-interpolate
        - vector : add to data, keep affine the same, re-interpolate
        - interpolated : must be same interpolation!
        """
        if isinstance(other, InterpolatedBaseRepresentationOrDifferential):
            if not array_equal(other.affine, self.affine):
                raise ValueError(_op_msg.format("subtract", self._class_))
        # Subtract, then re-interpolate
        return self._realize_class(self.data.__sub__(other), self.affine)

    def __mul__(
        self: IRoDType,
        other: T.Union[IRoDType, r.BaseRepresentationOrDifferential],
    ) -> IRoDType:
        """Add other to an InterpolatedBaseRepresentationOrDifferential

        If other:
        - point : add to data, keep affine the same, re-interpolate
        - vector : add to data, keep affine the same, re-interpolate
        - interpolated : must be same interpolation!
        """
        if isinstance(other, InterpolatedBaseRepresentationOrDifferential):
            if not array_equal(other.affine, self.affine):
                raise ValueError(_op_msg.format("multiply", self._class_))
        # Multiply, then re-interpolate
        return self._realize_class(self.data.__mul__(other), self.affine)

    def __truediv__(
        self: IRoDType,
        other: T.Union[IRoDType, r.BaseRepresentationOrDifferential],
    ) -> IRoDType:
        """Add other to an InterpolatedBaseRepresentationOrDifferential

        If other:
        - point : add to data, keep affine the same, re-interpolate
        - vector : add to data, keep affine the same, re-interpolate
        - interpolated : must be same interpolation!
        """
        if isinstance(other, InterpolatedBaseRepresentationOrDifferential):
            if not array_equal(other.affine, self.affine):
                raise ValueError(_op_msg.format("divide", self._class_))
        # Divide, then re-interpolate
        return self._realize_class(self.data.__truediv__(other), self.affine)

    # def _apply(self, method, *args, **kwargs):  # TODO!

    # ---------------------------------------------------------------
    # Specific wrappers

    def from_cartesian(
        self: IRoDType,
        other: T.Union[r.CartesianRepresentation, r.CartesianDifferential],
    ) -> IRoDType:
        """Create a representation of this class from a Cartesian one.

        Parameters
        ----------
        other : `CartesianRepresentation` or `CartesianDifferential`
            The representation to turn into this class

            Note: the affine parameter of this class is used. The
            representation must be the same length as the affine parameter.

        Returns
        -------
        representation : object of this class
            A new representation of this class's type.

        Raises
        ------
        ValueError
            If `other` is not same length as the this instance's affine
            parameter.
        """
        rep = self.data.from_cartesian(other)
        return self._class_(rep, self.affine, **self._interp_kwargs)

    # TODO just wrap self.data method with a wrapper?
    def to_cartesian(self: IRoDType) -> IRoDType:
        """Convert the representation to its Cartesian form.

        Note that any differentials get dropped. Also note that orientation
        information at the origin is *not* preserved by conversions through
        Cartesian coordinates. For example, transforming an angular position
        defined at distance=0 through cartesian coordinates and back will lose
        the original angular coordinates::

            >>> import astropy.units as u
            >>> import astropy.coordinates as coord
            >>> rep = coord.SphericalRepresentation(
            ...     lon=15*u.deg,
            ...     lat=-11*u.deg,
            ...     distance=0*u.pc)
            >>> rep.to_cartesian().represent_as(coord.SphericalRepresentation)
            <SphericalRepresentation (lon, lat, distance) in (rad, rad, pc)
                (0., 0., 0.)>

        Returns
        -------
        cartrepr : `CartesianRepresentation` or `CartesianDifferential`
            The representation in Cartesian form.
            If starting from a Cart
        """
        rep = self.data.to_cartesian()
        return self._class_(rep, self.affine, **self._interp_kwargs)

    def copy(self: IRoDType, *args: T.Any, **kwargs: T.Any) -> IRoDType:
        """Return an instance containing copies of the internal data.

        Parameters are as for :meth:`~numpy.ndarray.copy`.

        .. todo::

            this uses BaseRepresentation._apply, see if that may be modified
            instead

        Returns
        -------
        `interpolated_coordinates.InterpolatedBaseRepresentationOrDifferential`
            Same type as this instance.
        """
        data = self.data.copy(*args, **kwargs)
        interps = copy.deepcopy(self._interps)
        return self._class_(
            data,
            affine=self.affine,
            interps=interps,
            derivative_type=self.derivative_type,
            **self._interp_kwargs,
        )


# -------------------------------------------------------------------


class InterpolatedRepresentation(InterpolatedBaseRepresentationOrDifferential):
    """Wrapper for Representations, adding affine interpolations.

    .. todo::

        - override all the methods, mapping to underlying Representation
        - figure out how to do ``from_cartesian`` as a class method
        - get ``X_interp`` as properties. Need to do __dict__ manipulation,
          like BaseCoordinateFrame

    Parameters
    ----------
    representation : `~astropy.coordinates.BaseRepresentation` instance
    affine : `~astropy.units.Quantity` array-like
        The affine interpolation parameter.

    interps : Mapping or None (optional, keyword-only)
        Has same structure as a Representation

        .. code-block:: text

            dict(component name: interpolation,
                 ...
                 "differentials": dict(
                     "s" : dict(component name: interpolation,
                                ...),
                     ...))

    **interp_kwargs
        Only used if `interps` is None.
        keyword arguments into interpolation class

    Other Parameters
    ----------------
    interp_cls : Callable (optional, keyword-only)
        option for 'interp_kwargs'.
        If not specified, default is `IUSU`.
    """

    def __new__(
        cls: T.Type[IRType], representation: r.BaseRepresentation, *args: T.Any, **kwargs: T.Any
    ) -> T.Union[IRType, InterpolatedCartesianRepresentation]:
        self: T.Union[IRType, InterpolatedCartesianRepresentation]
        # Need to special case Cartesian b/c it has different methods
        if isinstance(representation, r.CartesianRepresentation):
            ccls = InterpolatedCartesianRepresentation
            self = super().__new__(ccls, representation, *args, **kwargs)
        else:
            self = super().__new__(cls, representation, *args, **kwargs)

        return self

    def __call__(self, affine: T.Optional[u.Quantity] = None) -> r.BaseRepresentation:
        """Evaluate interpolated representation.

        Parameters
        ----------
        affine : `~astropy.units.Quantity` array-like
            The affine interpolation parameter.
            If None, returns representation points.

        Returns
        -------
        :class:`~astropy.coordinates.BaseRepresenation`
            Representation of type ``self.data`` evaluated with `affine`
        """
        if affine is None:  # If None, returns representation as-is.
            return self.data

        affine = u.Quantity(affine, copy=False)  # need to ensure Quantity

        differentials = dict()
        for k, dif in self.data.differentials.items():
            differentials[k] = dif(affine)

        # evaluate the spline on each argument of the position
        params = {n: interp(affine) for n, interp in self._interps.items()}

        return self.data.__class__(**params, differentials=differentials)

    # ---------------------------------------------------------------

    # TODO just wrap self.data method with a wrapper?
    def represent_as(
        self: IRType,
        other_class: r.BaseRepresentation,
        differential_class: T.Optional[r.BaseDifferential] = None,
    ) -> InterpolatedRepresentation:
        """Convert coordinates to another representation.

        If the instance is of the requested class, it is returned unmodified.
        By default, conversion is done via Cartesian coordinates. Also note
        that orientation information at the origin is *not* preserved by
        conversions through Cartesian coordinates. See the docstring for
        `~astropy.coordinates.BaseRepresentation.represent_as()` for an
        example.

        Parameters
        ----------
        other_class : `~astropy.coordinates.BaseRepresentation` subclass
            The type of representation to turn the coordinates into.
        differential_class : dict of `~astropy.coordinates.BaseDifferential`, optional
            Classes in which the differentials should be represented.
            Can be a single class if only a single differential is attached,
            otherwise it should be a `dict` keyed by the same keys as the
            differentials.
        """
        rep = self.data.represent_as(other_class, differential_class=differential_class)

        # don't pass on the derivative_type
        # can't do self._class_ since InterpolatedCartesianRepresentation
        # only accepts `rep` of Cartesian type.
        return InterpolatedRepresentation(rep, self.affine, **self._interp_kwargs)

    # TODO just wrap self.data method with a wrapper?
    def with_differentials(self: IRType, differentials: T.Sequence[r.BaseDifferential]) -> IRType:
        """Realize Representation, with new differentials.

        Create a new representation with the same positions as this
        representation, but with these new differentials.

        Differential keys that already exist in this object's differential dict
        are overwritten.

        Parameters
        ----------
        differentials : Sequence of `~astropy.coordinates.BaseDifferential`
            The differentials for the new representation to have.

        Returns
        -------
        newrepr
            A copy of this representation, but with the ``differentials`` as
            its differentials.
        """
        if not differentials:  # (from source code)
            return self

        rep = self.data.with_differentials(differentials)
        return self._realize_class(rep, self.affine)

    # TODO just wrap self.data method with a wrapper?
    def without_differentials(self: IRType) -> IRType:
        """Return a copy of the representation without attached differentials.

        Returns
        -------
        newrepr
            A shallow copy of this representation, without any differentials.
            If no differentials were present, no copy is made.
        """
        if not self._differentials:  # from source code
            return self

        rep = self.data.without_differentials()
        return self._realize_class(rep, self.affine)

    def derivative(self: IRType, n: int = 1) -> InterpolatedDifferential:
        r"""Construct a new spline representing the derivative of this spline.

        Parameters
        ----------
        n : int, optional
            Order of derivative to evaluate. Default: 1
        """
        ideriv: InterpolatedDifferential
        if f"affine {n}" in self._derivatives:
            ideriv = self._derivatives[f"affine {n}"]
            return ideriv

        ideriv = super().derivative(n=n)
        self._derivatives[f"affine {n}"] = ideriv  # cache in derivatives

        return ideriv

    # ---------------------------------------------------------------
    # Convenience interpolation methods

    def headless_tangent_vectors(self: IRType) -> IRType:
        r"""Headless tangent vector at each point in affine.

        :math:`\vec{x} + \partial_{\affine} \vec{x}(\affine) \Delta\affine`

        .. todo::

            allow for passing my own points
        """
        irep = self.represent_as(r.CartesianRepresentation)
        ideriv = irep.derivative(n=1)  # (interpolated)

        offset = r.CartesianRepresentation(*(ideriv.d_xyz * self.affine.unit))
        offset = offset.represent_as(self.__class__)  # transform back

        return self._realize_class(offset, self.affine)

    def tangent_vectors(self: IRType) -> IRType:
        r"""Tangent vectors along the curve, from the origin.

        :math:`\vec{x} + \partial_{\affine} \vec{x}(\affine) \Delta\affine`

        .. todo::

            allow for passing my own points
        """
        irep = self.represent_as(r.CartesianRepresentation)
        ideriv = irep.derivative(n=1)  # (interpolated)

        offset = r.CartesianRepresentation(*(ideriv.d_xyz * self.affine.unit))

        newirep = irep + offset
        newirep = newirep.represent_as(self.__class__)

        return self._realize_class(newirep, self.affine)


class InterpolatedCartesianRepresentation(InterpolatedRepresentation):
    def __init__(
        self,
        rep: r.CartesianRepresentation,
        affine: T.Optional[u.Quantity],
        *,
        interps: T.Optional[T.Dict] = None,
        derivative_type: T.Optional[r.BaseDifferential] = None,
        **interp_kwargs: T.Any,
    ) -> None:

        # Check its instantiated and right class
        if inspect.isclass(rep) and issubclass(
            rep,
            r.CartesianRepresentation,
        ):
            raise ValueError("Must instantiate `rep`.")
        elif not isinstance(rep, r.CartesianRepresentation):
            raise TypeError("`rep` must be a `CartesianRepresentation`.")

        return super().__init__(
            rep,
            affine=affine,
            interps=interps,
            derivative_type=derivative_type,
            **interp_kwargs,
        )

    # TODO just wrap self.data method with a wrapper?
    def transform(self: ICRType, matrix: np.ndarray) -> ICRType:
        """Transform the cartesian coordinates using a 3x3 matrix.

        This returns a new representation and does not modify the original one.
        Any differentials attached to this representation will also be
        transformed.

        Parameters
        ----------
        matrix : `~numpy.ndarray`
            A 3x3 transformation matrix, such as a rotation matrix.


        Examples
        --------

        We can start off by creating a Cartesian representation object:

            >>> from astropy import units as u
            >>> from astropy.coordinates import CartesianRepresentation
            >>> rep = CartesianRepresentation([1, 2] * u.pc,
            ...                               [2, 3] * u.pc,
            ...                               [3, 4] * u.pc)

        We now create a rotation matrix around the z axis:

            >>> from astropy.coordinates.matrix_utilities import (
            ...     rotation_matrix)
            >>> rotation = rotation_matrix(30 * u.deg, axis='z')

        Finally, we can apply this transformation:

            >>> rep_new = rep.transform(rotation)
            >>> rep_new.xyz  # doctest: +FLOAT_CMP
            <Quantity [[ 1.8660254 , 3.23205081],
                       [ 1.23205081, 1.59807621],
                       [ 3.        , 4.        ]] pc>
        """
        return self._realize_class(self.data.transform(matrix), self.affine)

    def _scale_operation(self: ICRType, op: T.Callable, *args: T.Any) -> ICRType:  # type: ignore
        return self._realize_class(self.data._scale_operation(op, *args), self.affine)


# ===================================================================


class InterpolatedDifferential(InterpolatedBaseRepresentationOrDifferential):
    def __new__(
        cls: T.Type[IDType], rep: T.Union[IDType, DType], *args: T.Any, **kwargs: T.Any
    ) -> IDType:
        if not isinstance(rep, (InterpolatedDifferential, r.BaseDifferential)):
            raise TypeError("`rep` must be a differential type.")
        return super().__new__(cls, rep, *args, **kwargs)

    # ---------------------------------------------------------------

    def __call__(self, affine: T.Optional[u.Quantity] = None) -> DType:
        """Evaluate interpolated representation.

        Parameters
        ----------
        affine : `~astropy.units.Quantity` array-like
            The affine interpolation parameter.
            If None, returns representation points.

        Returns
        -------
        `~astropy.coordinates.BaseDifferential`
            Representation of type ``self.data`` evaluated with `affine`
        """
        data: DType

        if affine is None:  # If None, returns representation as-is.
            data = self.data
            return data

        # evaluate the spline on each argument of the position
        affine = u.Quantity(affine, copy=False)  # need to ensure Quantity
        params = {n: interp(affine) for n, interp in self._interps.items()}
        data = self.data.__class__(**params)
        return data

    # ---------------------------------------------------------------

    # TODO just wrap self.data method with a wrapper?
    def represent_as(
        self: IDType,
        other_class: r.BaseDifferential,
        base: r.BaseRepresentation,
    ) -> IDType:
        """Convert coordinates to another representation.

        If the instance is of the requested class, it is returned unmodified.
        By default, conversion is done via cartesian coordinates.

        Parameters
        ----------
        other_class : `~astropy.coordinates.BaseDifferential` subclass
            The type of representation to turn the coordinates into.
        base : instance of ``self.base_representation``
            Base relative to which the differentials are defined.  If the other
            class is a differential representation, the base will be converted
            to its ``base_representation``.
        """
        rep = self.data.represent_as(other_class, base=base)

        # don't pass on the derivative_type
        return self._class_(rep, self.affine, **self._interp_kwargs)

    def to_cartesian(self) -> InterpolatedCartesianRepresentation:  # type: ignore
        """Convert the differential to its Cartesian form.

        Note that any differentials get dropped. Also note that orientation
        information at the origin is *not* preserved by conversions through
        Cartesian coordinates. For example, transforming an angular position
        defined at distance=0 through cartesian coordinates and back will lose
        the original angular ccoordinates::

            >>> import astropy.units as u
            >>> import astropy.coordinates as coord
            >>> rep = coord.SphericalRepresentation(
            ...     lon=15*u.deg,
            ...     lat=-11*u.deg,
            ...     distance=0*u.pc)
            >>> rep.to_cartesian().represent_as(coord.SphericalRepresentation)
            <SphericalRepresentation (lon, lat, distance) in (rad, rad, pc)
                (0., 0., 0.)>

        Returns
        -------
        `CartesianRepresentation`
            The representation in Cartesian form.
            On Differentials, ``to_cartesian`` returns a Representation
            https://github.com/astropy/astropy/issues/6215
        """
        rep: r.CartesianRepresentation = self.data.to_cartesian()
        return InterpolatedCartesianRepresentation(rep, self.affine, **self._interp_kwargs)
