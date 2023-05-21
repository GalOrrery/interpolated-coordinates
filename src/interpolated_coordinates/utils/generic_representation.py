r"""Generic versions of the |Representation| in :mod:`astropy`.

These classes assume less about the constituent dimensions than their astropy
counterparts. This can be useful for working with phase-spaces that are not
real-space positions (or derivates thereof). However, care should be taken when
using many of the methods of these generic representations since they inherit
from the astropy real-space representation machinery.

You will probably not be instantiating these classes directly, but encountering
them from :mod:`interpolated_coordinates` classes like
:class:`interpolated_coordinates.InterpolatedSkyCoord`. However, one can get and
use the classes, with all the above-noted caveats:

    >>> import astropy.units as u
    >>> from interpolated_coordinates.utils.generic_representation import \
    ...     GenericCartesianRepresentation

    >>> r = GenericCartesianRepresentation(1, 2, 3)
    >>> r
    <GenericCartesianRepresentation (x, y, z) [dimensionless]
        (1., 2., 3.)>
    >>> r.x
    <Quantity 1.>

The real convenience lies with differentials, which can go to arbitrary order.
Recalling that many of the methods will give incorrect results, these classes
are primarily useful for consistent and familiar data storage.

    >>> from interpolated_coordinates.utils.generic_representation import \
    ...     GenericSpherical2ndDifferential

    >>> d2 = GenericSpherical2ndDifferential(1 * u.rad/u.s**2, 2 * u.rad/u.s**2, 3 * u.km/u.s**2)
    >>> d2
    <GenericSpherical2ndDifferential (d_lon, d_lat, d_distance) in (rad / s2, rad / s2, km / s2)
        (1., 2., 3.)>
"""

##############################################################################
# IMPORTS

from __future__ import annotations

import inspect
import sys
from functools import reduce
from typing import Any, cast

import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates.representation import DIFFERENTIAL_CLASSES

__all__ = [
    "GenericRepresentation",
    "GenericDifferential",
]

##############################################################################
# PARAMETERS

_GENERIC_REGISTRY: dict[
    object | str,
    GenericRepresentation | GenericDifferential,
] = {}


##############################################################################
# CODE
##############################################################################


class GenericRepresentationOrDifferential(coord.BaseRepresentationOrDifferential):
    pass


class GenericRepresentation(coord.BaseRepresentation, GenericRepresentationOrDifferential):
    """Generic representation of a point in a 3D coordinate system.

    Parameters
    ----------
    q1, q2, q3 : `~astropy.units.Quantity` or subclass
        The components of the 3D points. The names are the keys and the
        subclasses the values of the ``attr_classes`` attribute.

    differentials : dict, `~astropy.coordinates.BaseDifferential`, optional
        Any differential classes that should be associated with this
        representation. The input must either be a single
        `~astropy.coordinates.BaseDifferential` subclass instance, or a
        dictionary with keys set to a string representation of the SI unit
        with which the differential (derivative) is taken. For example, for a
        velocity differential on a positional representation, the key would be
        ``'s'`` for seconds, indicating that the derivative is a time
        derivative.
    copy : bool, optional
        If `True` (default), arrays will be copied. If `False`, arrays will
        be references, though possibly broadcast to ensure matching shapes.

    Notes
    -----
    All representation classes should subclass this base representation class,
    and define an ``attr_classes`` attribute, an `~collections.OrderedDict`
    which maps component names to the class that creates them. They must also
    define a ``to_cartesian`` method and a ``from_cartesian`` class method. By
    default, transformations are done via the cartesian system, but classes
    that want to define a smarter transformation path can overload the
    ``represent_as`` method. If one wants to use an associated differential
    class, one should also define ``unit_vectors`` and ``scale_factors``
    methods (see those methods for details).
    """

    attr_classes = {"q1": u.Quantity, "q2": u.Quantity, "q3": u.Quantity}

    @staticmethod
    def _make_generic_cls(
        rep_cls: coord.BaseRepresentation | GenericRepresentation,
    ) -> GenericRepresentation:
        """Return a generic form of a representation.

        Parameters
        ----------
        rep_cls : |Representation| or `GenericRepresentation`
            Representation class for which to make generic.

        Returns
        -------
        `GenericRepresentation` subclass
            Generic form of `rep_cls`.
            If `rep_cls` is already generic, return it unchanged.
            Subclasses are cached in a registry.
        """
        cls: GenericRepresentation

        # 1) Check if it's already generic
        if issubclass(rep_cls, GenericRepresentation):
            cls = rep_cls

        # 2) Check if it's cached
        elif rep_cls in _GENERIC_REGISTRY:
            cls = _GENERIC_REGISTRY[rep_cls]

        # 3) Need to dynamically define the generic class
        else:
            name = f"Generic{rep_cls.__name__}"
            bases = (GenericRepresentation, rep_cls)

            # attributes: copies `attr_classes`
            attrs_meths = {"attr_classes": rep_cls.attr_classes}
            # add link from `qX` to the attr method  # TODO!
            # for i, k in enumerate(rep_cls.attr_classes.keys()):
            #     def get_attr(self):

            cls = cast(GenericRepresentation, type(name, bases, attrs_meths))

            # cache b/c can only define the same Rep/Dif once
            _GENERIC_REGISTRY[rep_cls] = cls

            # also store in locals
            setattr(sys.modules[__name__], cls.__name__, cls)
            sys.modules[__name__].__all__.append(cls.__name__)

        return cls


# -------------------------------------------------------------------


def _ordinal(n: int) -> str:
    """Return suffix for ordinal.

    Credits: https://codegolf.stackexchange.com/a/74047

    Parameters
    ----------
    n : int
        Must be >= 1

    Returns
    -------
    str
        Ordinal form `n`. Ex 1 -> '1st', 2 -> '2nd', 3 -> '3rd'.
    """
    i: int = n % 5 * (n % 100 ^ 15 > 4 > n % 10)  # noqa: PLR2004
    return str(n) + "tsnrhtdd"[i::4]


class GenericDifferential(coord.BaseDifferential, GenericRepresentationOrDifferential):
    r"""A base class representing differentials of representations.

    These represent differences or derivatives along each component.
    E.g., for physics spherical coordinates, these would be
    :math:`\delta r, \delta \theta, \delta \phi`.

    Parameters
    ----------
    d_q1, d_q2, d_q3 : `~astropy.units.Quantity` or subclass
        The components of the 3D differentials.  The names are the keys and the
        subclasses the values of the ``attr_classes`` attribute.
    copy : bool, optional
        If `True` (default), arrays will be copied. If `False`, arrays will
        be references, though possibly broadcast to ensure matching shapes.
    """

    base_representation: coord.BaseRepresentation = GenericRepresentation

    @staticmethod
    def _make_generic_cls(
        dif_cls: coord.BaseDifferential | GenericDifferential,
        n: int = 1,
    ) -> GenericDifferential:
        """Make Generic Differential.

        Parameters
        ----------
        dif_cls : |Differential| or `GenericDifferential` class
            Differential class for which to make generic.
        n : int
            The differential level.
            Not used if `dif_cls` is `GenericDifferential`

        Returns
        -------
        `GenericDifferential`
            Generic form of `dif_cls`.
            If `dif_cls` is already generic, return it unchanged.
            Subclasses are cached in a registry.
        """
        # 1) check if it's already generic
        if issubclass(dif_cls, GenericDifferential):
            return dif_cls

        # 2) check if `n` is too small to make a differential
        if n < 1:
            msg = "n < 1"
            raise ValueError(msg)

        # 3) make name for generic.
        if n == 1:  # a) special case for n=1
            name = f"Generic{dif_cls.__name__}"
        else:  # b) higher ordinal
            dif_type = dif_cls.__name__[: -len("Differential")]
            name = f"Generic{dif_type}{_ordinal(n)}Differential"

        cls: GenericDifferential

        # A) check if cached
        if n == 1 and dif_cls in _GENERIC_REGISTRY:  # i) special case for n=1
            cls = _GENERIC_REGISTRY[dif_cls]
        elif name in _GENERIC_REGISTRY:  # ii) higher ordinal
            cls = _GENERIC_REGISTRY[name]

        # B) make generic
        else:
            bases = (GenericDifferential, dif_cls)

            # get base representation from differential class.
            # and then get the generic form
            generic_base = GenericRepresentation._make_generic_cls(
                dif_cls.base_representation,
            )

            # attributes: copies `attr_classes`
            attrs_meths = {
                "attr_classes": dif_cls.attr_classes,
                "base_representation": generic_base,
            }

            # Make generic differential
            cls = cast(GenericDifferential, type(name, bases, attrs_meths))

            # cache, either by class or by name
            _GENERIC_REGISTRY[dif_cls if n == 1 else name] = cls

            # also store in locals
            setattr(sys.modules[__name__], cls.__name__, cls)
            sys.modules[__name__].__all__.append(cls.__name__)

        return cls

    @staticmethod
    def _make_generic_cls_for_representation(
        rep_cls: coord.BaseRepresentation,
        n: int = 1,
    ) -> GenericDifferential:
        """Make generic differential given a representation.

        Parameters
        ----------
        rep_cls : |Representation| class
            Representation class for which to make generic.
        n : int
            Must be >= 1

        Returns
        -------
        `GenericDifferential`
            Of ordinal `n`
        """
        rep_cls_name: str = rep_cls.__name__[: -len("Representation")]

        if n == 1:
            name = f"Generic{rep_cls_name}Differential"
        else:
            name = f"Generic{rep_cls_name}{_ordinal(n)}Differential"

        cls: GenericDifferential

        if name in _GENERIC_REGISTRY:
            cls = _GENERIC_REGISTRY[name]
        elif dcls := DIFFERENTIAL_CLASSES.get(rep_cls_name.lower()):
            cls = GenericDifferential._make_generic_cls(dcls, n=n)

        else:
            cls = cast(
                GenericDifferential,
                type(
                    name,
                    (GenericDifferential, rep_cls),
                    {"base_representation": rep_cls},
                ),
            )

            _GENERIC_REGISTRY[name] = cls

            # also store in locals
            setattr(sys.modules[__name__], cls.__name__, cls)
            sys.modules[__name__].__all__.append(cls.__name__)

        return cls


# ===================================================================


def __getattr__(name: str) -> type:
    """Get a generic direct subclass of an Astropy representation.

    Parameters
    ----------
    name : str
        Name of the class.

    Returns
    -------
    type

    Raises
    ------
    AttributeError
        If `name` doesn't start with "Generic", the "Generic"-removed name is
        not for a :class:`~astropy.coordinates.BaseRepresentation` or
        :class:`~astropy.coordinates.BaseDifferential`.
    """
    if name.startswith("Generic"):
        name = name.replace("Generic", "")
    if name.endswith("Differential"):  # strip the ordinal
        # Get the type, e.g. Cartesian
        i: int = len("Differential")
        kind: str = reduce(lambda k, n: k.split(n)[0], "0123456789", name[:-i])
        # Get order of the differential
        j: int = len(kind)
        order: str = reduce(lambda o, s: o.split(s)[0], "tsnrhd", name[j:-i])
        n: int = int(order) if order else 1  # 1st derivative is an empty string

        name = kind + "Differential"

    cls: coord.RepresentationOrDifferential | Any
    if cls := getattr(coord, name, False):
        generic_cls: GenericRepresentationOrDifferential
        if inspect.isclass(cls) and issubclass(cls, coord.BaseRepresentation):
            generic_cls = GenericRepresentation._make_generic_cls(cls)
        elif inspect.isclass(cls) and issubclass(cls, coord.BaseDifferential):
            generic_cls = GenericDifferential._make_generic_cls(cls, n=n)
        else:
            msg = f"module {__name__!r} has no attribute {name!r}"
            raise AttributeError(msg)

        return generic_cls

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


# def __dir__():
#     return sorted(dir_out + __all__ + )
