"""Interpolated representations."""

from __future__ import annotations

__all__ = [
    "InterpolatedBaseRepresentationOrDifferential",
    "InterpolatedRepresentation",
    "InterpolatedCartesianRepresentation",
    "InterpolatedDifferential",
]

import abc
import copy
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import astropy.units as u
import numpy.lib.recfunctions as rfn
from astropy.coordinates import (
    BaseDifferential,
    BaseRepresentation,
    BaseRepresentationOrDifferential,
    CartesianDifferential,
    CartesianRepresentation,
    RadialDifferential,
    UnitSphericalCosLatDifferential,
    UnitSphericalDifferential,
)
from numpy import array2string, array_equal

from .utils import GenericDifferential
from .utils import InterpolatedUnivariateSplinewithUnits as IntpUnivarSplUnits

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from ._type_hints import UnitLikeType

##############################################################################
# PARAMETERS

# the unit-differentials
_UNIT_DIF_TYPES: tuple[type[BaseDifferential], ...] = (
    UnitSphericalDifferential,
    UnitSphericalCosLatDifferential,
    RadialDifferential,
)

IRoDType = TypeVar("IRoDType", bound="InterpolatedBaseRepresentationOrDifferential")
IRType = TypeVar("IRType", bound="InterpolatedRepresentation")
ICRType = TypeVar("ICRType", bound="InterpolatedCartesianRepresentation")
IDType = TypeVar("IDType", bound="InterpolatedDifferential")
DType = TypeVar("DType", bound=BaseDifferential)


_op_msg = "can only {} {} if the interpolation variables are the same."


##############################################################################
# CODE
##############################################################################


def _find_first_best_compatible_differential(
    rep: BaseRepresentation,
    n: int = 1,
) -> BaseDifferential | GenericDifferential:
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
    #         rep.__class__,

    if n != 1:
        derivative_type = GenericDifferential._make_generic_cls(derivative_type, n=n)

    return derivative_type


def _infer_derivative_type(
    rep: BaseRepresentationOrDifferential,
    dif_unit: UnitLikeType,
    n: int = 1,
) -> BaseDifferential | GenericDifferential:
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
    `astropy.coordinates.BaseDifferential` | `~interpolated_coordinates.utils.GenericDifferential`

    """
    unit = u.Unit(dif_unit)
    rep_cls = rep.__class__  # (store rep class for line length)

    # Now check we can even do this: if can't make a better Generic
    # 1) can't for `Differentials` and stuff without compatible diffs
    if isinstance(rep, BaseDifferential):
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
    rep : `~astropy.coordinates.BaseRepresentation` instance, positional-only
    affine : `~astropy.units.Quantity` array-like
        The affine interpolation parameter.

    interps : Mapping or None, optional keyword-only
        Has same structure as a Representation

        .. code-block:: text

            dict(component name: interpolation,
                ...
                "differentials": dict(
                    "s" : dict(component name: interpolation, ...),
                    ...))

    **interp_kwargs : Any
        Only used if `interps` is None.
        keyword arguments into interpolation class

    Other Parameters
    ----------------
    interp_cls : Callable (optional, keyword-only)
        option for 'interp_kwargs'.
        If not specified, default is `IntpUnivarSplUnits`.

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

    def __new__(cls: type[IRoDType], *_: Any, **__: Any) -> IRoDType:
        if cls is InterpolatedBaseRepresentationOrDifferential:
            msg = f"Cannot instantiate a {cls}."
            raise TypeError(msg)

        inst: IRoDType = super().__new__(cls)
        return inst

    def __init__(
        self,
        rep: BaseRepresentationOrDifferential,
        /,
        affine: u.Quantity,
        *,
        interps: dict | None = None,
        derivative_type: BaseDifferential | None = None,
        **interp_kwargs: Any,
    ) -> None:
        # Check it's instantiated and right class
        if not isinstance(rep, BaseRepresentationOrDifferential):
            msg = "`rep` must be a `BaseRepresentationOrDifferential`."
            raise TypeError(msg)

        # Affine parameter
        affine = u.Quantity(affine, copy=False)  # ensure Quantity
        if affine.ndim != 1:
            msg = "`affine` must be 1-D."
            raise ValueError(msg)
        if len(affine) != len(rep):
            msg = "`affine` must be same length as `rep`"
            raise ValueError(msg)

        # store representation and affine parameter
        self.data = rep
        self._affine = affine = u.Quantity(affine, copy=True)  # TODO copy?

        # The class to use when differentiating wrt to the affine parameter.
        if derivative_type is None:
            derivative_type = _infer_derivative_type(rep, affine.unit)
        # TODO better detection if derivative_type doesn't work!
        self._derivative_type = derivative_type
        self._derivatives: dict[str, Any] = {}

        self._init_interps(rep, affine, interps, interp_kwargs)  # Construct interpolation

    def _init_interps(
        self,
        rep: BaseRepresentationOrDifferential,
        affine: u.Quantity,
        interps: dict | None,
        interp_kwargs: dict[str, Any],
    ) -> None:
        self._interp_kwargs = interp_kwargs.copy()  # TODO need copy?

        if interps is not None:
            self._interps = interps
            return

        self._interps = {}

        # determine interpolation type
        interp_cls = interp_kwargs.pop("interp_cls", IntpUnivarSplUnits)

        # Positional information
        for comp in rep.components:
            self._interps[comp] = interp_cls(affine, getattr(rep, comp), **interp_kwargs)

        # differentials information
        # these are stored in a dictionary with keys wrt time
        # ex : rep.differentials["s"] is a Velocity
        for k, differential in getattr(rep, "differentials", {}).items():
            d_derivative_type: type | None

            # Is this already an InterpolatedDifferential?
            # then need to pop back to the Differential
            if isinstance(differential, InterpolatedDifferential):
                d_derivative_type = differential.derivative_type
                differential = differential.data  # noqa: PLW2901
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
    def _class_(self: IRoDType) -> type[IRoDType]:
        """Get this object's true class, not the un-interpolated class."""
        return type(self)

    def _realize_class(self: IRoDType, rep: BaseRepresentation, affine: u.Quantity) -> IRoDType:
        inst: IRoDType = self._class_(
            rep,
            affine,
            derivative_type=self.derivative_type,
            **self._interp_kwargs,
        )
        return inst

    #################################################################
    # Interpolation Methods

    @abc.abstractmethod
    def __call__(self, affine: u.Quantity | None = None) -> IRoDType:
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

        return ideriv  # noqa: RET504

    # def antiderivative(self, n: int = 1) -> Any:
    #     r"""Construct a new spline representing the integral of this spline.
    #
    #     Parameters
    #     ----------
    #     n : int, optional
    #         Order of derivative to evaluate. Default: 1
    #     """
    #     # evaluate the spline on each argument of the position
    #         for k, interp in self._interps.items()

    # def integral(self, a, b):
    #     """Return definite integral between two given points."""

    #################################################################
    # Mapping to Underlying Representation

    @property
    def uninterpolated(self) -> BaseRepresentationOrDifferential:
        """Return the underlying Representation."""
        return self.data

    # ---------------------------------------------------------------
    # Hidden methods

    @property
    def __class__(self) -> type[BaseRepresentationOrDifferential]:
        """Make class appear the same as the underlying Representation."""
        cls: type[BaseRepresentationOrDifferential] = self.data.__class__
        return cls

    @__class__.setter
    def __class__(self, value: Any) -> None:  # needed for mypy
        msg = "cannot set attribute ``__class__``."
        raise TypeError(msg)

    def __getattr__(self, key: str) -> Any:
        """Route everything to underlying Representation."""
        return getattr(object.__getattribute__(self, "data"), key)

    def __getitem__(self: IRoDType, key: str | slice | NDArray) -> IRoDType:
        """Getitem on Representation, re-interpolating."""
        rep: BaseRepresentation = self.data[key]
        afn: u.Quantity = self.affine[key]
        inst: IRoDType = self._realize_class(rep, afn)
        return inst

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        """String representation, adding interpolation information."""
        prefixstr = "    "
        values = rfn.merge_arrays((self.affine.value, self.data._values), flatten=True)
        arrstr = array2string(values, prefix=prefixstr, separator=", ")

        diffstr = ""
        if getattr(self, "differentials", None):
            diffstr = "\n (has differentials w.r.t.: {})".format(
                ", ".join([repr(key) for key in self.differentials]),
            )

        aurep = str(self.affine.unit) or "[dimensionless]"

        _ustr = self.data._unitstr
        if _ustr:
            unitstr = f"in ({aurep}| {_ustr[1:]}" if _ustr[0] == "(" else f"in {aurep}| {_ustr}"
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
        self: IRoDType,
        op: Callable,
        *args: Any,
        scaled_base: bool = False,
    ) -> IRoDType:
        rep = self.data._scale_operation(op, *args, scaled_base=scaled_base)
        inst: IRoDType = self._realize_class(rep, self.affine)
        return inst

    # ---------------------------------------------------------------
    # Math Methods

    def __add__(
        self: IRoDType,
        other: BaseRepresentationOrDifferential | IRoDType,
    ) -> IRoDType:
        """Add other to an InterpolatedBaseRepresentationOrDifferential.

        If other:
        - point : add to data, keep affine the same, re-interpolate
        - vector : add to data, keep affine the same, re-interpolate
        - interpolated : must be same interpolation!
        """
        if isinstance(other, InterpolatedBaseRepresentationOrDifferential) and not array_equal(
            other.affine,
            self.affine,
        ):
            raise ValueError(_op_msg.format("add", self._class_))
        # Add, then re-interpolate
        return self._realize_class(self.data.__add__(other), self.affine)

    def __sub__(
        self: IRoDType,
        other: IRoDType | BaseRepresentationOrDifferential,
    ) -> IRoDType:
        """Add other to an InterpolatedBaseRepresentationOrDifferential.

        If other:
        - point : add to data, keep affine the same, re-interpolate
        - vector : add to data, keep affine the same, re-interpolate
        - interpolated : must be same interpolation!
        """
        if isinstance(other, InterpolatedBaseRepresentationOrDifferential) and not array_equal(
            other.affine,
            self.affine,
        ):
            raise ValueError(_op_msg.format("subtract", self._class_))
        # Subtract, then re-interpolate
        return self._realize_class(self.data.__sub__(other), self.affine)

    def __mul__(
        self: IRoDType,
        other: IRoDType | BaseRepresentationOrDifferential,
    ) -> IRoDType:
        """Add other to an InterpolatedBaseRepresentationOrDifferential.

        If other:
        - point : add to data, keep affine the same, re-interpolate
        - vector : add to data, keep affine the same, re-interpolate
        - interpolated : must be same interpolation!
        """
        if isinstance(other, InterpolatedBaseRepresentationOrDifferential) and not array_equal(
            other.affine,
            self.affine,
        ):
            raise ValueError(_op_msg.format("multiply", self._class_))
        # Multiply, then re-interpolate
        return self._realize_class(self.data.__mul__(other), self.affine)

    def __truediv__(
        self: IRoDType,
        other: IRoDType | BaseRepresentationOrDifferential,
    ) -> IRoDType:
        """Add other to an InterpolatedBaseRepresentationOrDifferential.

        If other:
        - point : add to data, keep affine the same, re-interpolate
        - vector : add to data, keep affine the same, re-interpolate
        - interpolated : must be same interpolation!
        """
        if isinstance(other, InterpolatedBaseRepresentationOrDifferential) and not array_equal(
            other.affine,
            self.affine,
        ):
            raise ValueError(_op_msg.format("divide", self._class_))
        # Divide, then re-interpolate
        return self._realize_class(self.data.__truediv__(other), self.affine)

    # def _apply(self, method, *args, **kwargs):  # TODO!

    # ---------------------------------------------------------------
    # Specific wrappers

    def from_cartesian(
        self: IRoDType,
        other: CartesianRepresentation | CartesianDifferential,
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

    def copy(self: IRoDType, *args: Any, **kwargs: Any) -> IRoDType:
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
        args, kwargs = self.__getnewargs_ex__()
        return self._class_(data, *args[1:], **kwargs)

    def __getnewargs_ex__(self) -> tuple[tuple[Any, ...], dict[str, Any]]:
        args = (self.data, self.affine)
        kwargs = {
            "interps": copy.deepcopy(self._interps),
            "derivative_type": self.derivative_type,
            **self._interp_kwargs,
        }
        return args, kwargs


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
        If not specified, default is `IntpUnivarSplUnits`.

    """

    def __new__(
        cls: type[IRType],
        representation: BaseRepresentation,
        *args: Any,
        **kwargs: Any,
    ) -> IRType | InterpolatedCartesianRepresentation:
        self: IRType | InterpolatedCartesianRepresentation
        # Need to special case Cartesian b/c it has different methods
        if isinstance(representation, CartesianRepresentation):
            ccls = InterpolatedCartesianRepresentation
            self = super().__new__(ccls, representation, *args, **kwargs)
        else:
            self = super().__new__(cls, representation, *args, **kwargs)

        return self

    def __call__(self, affine: u.Quantity | None = None) -> BaseRepresentation:
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

        differentials = {}
        for k, dif in self.data.differentials.items():
            differentials[k] = dif(affine)

        # evaluate the spline on each argument of the position
        params = {n: interp(affine) for n, interp in self._interps.items()}

        return self.data.__class__(**params, differentials=differentials)

    @property
    def uninterpolated(self) -> BaseRepresentation:
        """Return the underlying Representation."""
        data = copy.deepcopy(self.data)
        data.diffentials = {k: dif.uninterpolated for k, dif in data.differentials.items()}
        return data

    # ---------------------------------------------------------------

    # TODO just wrap self.data method with a wrapper?
    def represent_as(
        self: IRType,
        other_class: BaseRepresentation,
        differential_class: BaseDifferential | None = None,
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
    def with_differentials(self: IRType, differentials: Sequence[BaseDifferential]) -> IRType:
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
            return self._derivatives[f"affine {n}"]

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
        irep = self.represent_as(CartesianRepresentation)
        ideriv = irep.derivative(n=1)  # (interpolated)

        offset = CartesianRepresentation(*(ideriv.d_xyz * self.affine.unit))
        offset = offset.represent_as(self.__class__)  # transform back

        return self._realize_class(offset, self.affine)

    def tangent_vectors(self: IRType) -> IRType:
        r"""Tangent vectors along the curve, from the origin.

        :math:`\vec{x} + \partial_{\affine} \vec{x}(\affine) \Delta\affine`

        .. todo::

            allow for passing my own points
        """
        irep = self.represent_as(CartesianRepresentation)
        ideriv = irep.derivative(n=1)  # (interpolated)

        offset = CartesianRepresentation(*(ideriv.d_xyz * self.affine.unit))

        newirep = irep + offset
        newirep = newirep.represent_as(self.__class__)

        return self._realize_class(newirep, self.affine)


class InterpolatedCartesianRepresentation(InterpolatedRepresentation):  # noqa: D101
    def __init__(
        self,
        rep: CartesianRepresentation,
        affine: u.Quantity | None,
        *,
        interps: dict | None = None,
        derivative_type: BaseDifferential | None = None,
        **interp_kwargs: Any,
    ) -> None:
        # Check its instantiated and right class
        if not isinstance(rep, CartesianRepresentation):
            msg = "`rep` must be a `CartesianRepresentation`."
            raise TypeError(msg)

        super().__init__(
            rep,
            affine=affine,
            interps=interps,
            derivative_type=derivative_type,
            **interp_kwargs,
        )

    # TODO just wrap self.data method with a wrapper?
    def transform(self: ICRType, matrix: NDArray) -> ICRType:
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

    def _scale_operation(self: ICRType, op: Callable, *args: Any) -> ICRType:
        return self._realize_class(
            self.data._scale_operation(op, *args),
            self.affine,
        )


# ===================================================================


class InterpolatedDifferential(InterpolatedBaseRepresentationOrDifferential):  # noqa: D101
    def __new__(cls: type[IDType], rep: IDType | DType, *args: Any, **kwargs: Any) -> IDType:
        if not isinstance(rep, (InterpolatedDifferential, BaseDifferential)):
            msg = "`rep` must be a differential type."
            raise TypeError(msg)
        return super().__new__(cls, rep, *args, **kwargs)

    # ---------------------------------------------------------------

    def __call__(self, affine: u.Quantity | None = None) -> DType:
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
        if affine is None:  # If None, returns representation as-is.
            return self.data

        # evaluate the spline on each argument of the position
        affine = u.Quantity(affine, copy=False)  # need to ensure Quantity
        params = {n: interp(affine) for n, interp in self._interps.items()}
        return self.data.__class__(**params)

    # ---------------------------------------------------------------

    # TODO just wrap self.data method with a wrapper?
    def represent_as(
        self: IDType,
        other_class: BaseDifferential,
        base: BaseRepresentation,
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

    def to_cartesian(self) -> InterpolatedCartesianRepresentation:
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
        rep: CartesianRepresentation = self.data.to_cartesian()
        return InterpolatedCartesianRepresentation(rep, self.affine, **self._interp_kwargs)
