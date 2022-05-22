# -*- coding: utf-8 -*-

from __future__ import annotations

# STDLIB
from typing import Any, Dict, List, Optional, Tuple, Type, Union

# THIRD PARTY
from astropy.coordinates import (
    Angle,
    BaseCoordinateFrame,
    BaseDifferential,
    BaseRepresentation,
    BaseRepresentationOrDifferential,
    Distance,
    SkyCoord,
)
from astropy.units import Quantity
from astropy.utils.decorators import format_doc
from numpy import ndarray

# LOCAL
from ._type_hints import CoordinateType, FrameLikeType, RepLikeType
from .representation import (
    _UNIT_DIF_TYPES,
    InterpolatedBaseRepresentationOrDifferential,
    InterpolatedRepresentation,
)
from .utils import InterpolatedUnivariateSplinewithUnits as IUSU

__all__ = ["InterpolatedCoordinateFrame", "InterpolatedSkyCoord"]


##############################################################################
# CODE
##############################################################################


class InterpolatedCoordinateFrame:
    """Wrapper for Coordinate Frame, adding affine interpolations.

    .. todo::

        - override all the methods, mapping to underlying CoordinateFrame

    Parameters
    ----------
    data : InterpolatedRepresentation or Representation or CoordinateFrame
        For either an InterpolatedRepresentation or Representation
        the kwarg 'frame' must also be specified.
        If CoordinateFrame, then 'frame' is ignored.
    affine : Quantity array-like (optional)
        if not a Quantity, one is assigned.
        Only used if data is not already interpolated.
        If data is NOT interpolated, this is required.


    Other Parameters
    ----------------
    frame : str or CoordinateFrame
        only used if `data` is  an InterpolatedRepresentation or Representation

    Raises
    ------
    Exception
        if `frame` has no error
    ValueError
        if `data` is not an interpolated type and `affine` is None
    TypeError
        if `data` is not one of types specified in Parameters.
    """

    def __init__(
        self,
        data: CoordinateType,
        affine: Optional[Quantity] = None,
        *,
        interps: Optional[Dict] = None,
        **interp_kwargs: Any,
    ) -> None:
        # get rep from CoordinateType
        rep = data.data

        if isinstance(rep, InterpolatedRepresentation):
            pass
        elif isinstance(rep, BaseRepresentation):
            if affine is None:
                raise ValueError(
                    "`data` is not already interpolated. "
                    "Need to pass a Quantity array for `affine`.",
                )

            rep = InterpolatedRepresentation(rep, affine=affine, interps=interps, **interp_kwargs)
        else:
            raise TypeError(
                "`data` must be type " + "<InterpolatedRepresentation> or <BaseRepresentation>",
            )

        self.frame = data.realize_frame(rep)
        self._interp_kwargs = interp_kwargs

    @property
    def _interp_kwargs(self) -> Dict[str, Any]:
        ikw: dict = self.data._interp_kwargs
        return ikw

    @_interp_kwargs.setter
    def _interp_kwargs(self, value: dict) -> None:
        self.data._interp_kwargs = value

    def __call__(self, affine: Optional[Quantity] = None) -> BaseRepresentation:
        """Evaluate interpolated coordinate frame.

        Parameters
        ----------
        affine : `~astropy.units.Quantity` array-like
            The affine interpolation parameter.
            If None, returns representation points.

        Returns
        -------
        BaseRepresenation
            Representation of type ``self.data`` evaluated with `affine`
        """
        return self.frame.realize_frame(self.frame.data(affine))

    @property
    def _class_(self) -> Type[InterpolatedCoordinateFrame]:
        return object.__class__(self)

    def _realize_class(self, data: CoordinateType) -> InterpolatedCoordinateFrame:
        return self._class_(data, affine=self.affine, **self._interp_kwargs)

    def realize_frame(
        self, data: BaseRepresentation, affine: Optional[Quantity] = None, **kwargs: Any
    ) -> InterpolatedCoordinateFrame:
        """Generates a new frame with new data from another frame (which may or
        may not have data). Roughly speaking, the converse of
        `replicate_without_data`.

        Parameters
        ----------
        data : `~astropy.coordinates.BaseRepresentation`
            The representation to use as the data for the new frame.

        Any additional keywords are treated as frame attributes to be set on
        the new frame object. In particular, `representation_type` can be
        specified.

        Returns
        -------
        frameobj : same as this frame
            A new object with the same frame attributes as this one, but
            with the ``data`` as the coordinate data.
        """
        frame = self.frame.realize_frame(data, **kwargs)
        return self._class_(frame, affine=affine, **kwargs)

    #################################################################
    # Interpolation Methods
    # Mapped to underlying Representation

    @format_doc(InterpolatedBaseRepresentationOrDifferential.derivative.__doc__)
    def derivative(self, n: int = 1) -> BaseRepresentationOrDifferential:
        """Take nth derivative wrt affine parameter."""
        return self.frame.data.derivative(n=n)

    @property
    def affine(self) -> Quantity:  # read-only
        return self.frame.data.affine

    def headless_tangent_vectors(self) -> InterpolatedCoordinateFrame:
        r"""Headless tangent vector at each point in affine.

        :math:`\vec{x} + \partial_{\affine} \vec{x}(\affine) \Delta\affine`

        .. todo::

            allow for passing my own points
        """
        rep = self.frame.data.headless_tangent_vectors()
        return self.realize_frame(rep)

    def tangent_vectors(self) -> InterpolatedCoordinateFrame:
        r"""Tangent vectors along the curve, from the origin.

        :math:`\vec{x} + \partial_{\affine} \vec{x}(\affine) \Delta\affine`

        .. todo::

            allow for passing my own points
        """
        rep = self.frame.data.tangent_vectors()
        return self.realize_frame(rep)

    #################################################################
    # Mapping to Underlying CoordinateFrame

    @property
    def __class__(self) -> type:
        """Make class appear the same as the underlying CoordinateFrame."""
        cls: type = self.frame.__class__
        return cls

    @__class__.setter
    def __class__(self, value: Any) -> None:  # needed for mypy  # noqa: F811
        raise TypeError("cannot set the `__class__` attribute.")

    def __getattr__(self, key: str) -> Any:
        """Route everything to underlying CoordinateFrame."""
        try:
            frame = object.__getattribute__(self, "frame")
        except AttributeError:
            raise
        else:
            return getattr(frame, key)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, key: Union[int, slice, ndarray]) -> InterpolatedCoordinateFrame:
        frame = self.frame[key]
        affine = self.affine[key]

        iframe = self._class_(frame, affine=affine, **self._interp_kwargs)
        iframe.representation_type = self.representation_type

        return iframe

    @property
    def representation_type(self) -> BaseRepresentation:
        return self.frame.representation_type

    @representation_type.setter
    def representation_type(self, value: BaseRepresentation) -> None:
        self.frame.representation_type = value

    def represent_as(
        self,
        base: RepLikeType,
        s: Union[str, BaseDifferential] = "base",
        in_frame_units: bool = False,
    ) -> InterpolatedRepresentation:
        """Generate and return a new representation of this frame's `data`
        as a Representation object.

        Note: In order to make an in-place change of the representation
        of a Frame or SkyCoord object, set the ``representation``
        attribute of that object to the desired new representation, or
        use the ``set_representation_cls`` method to also set the differential.

        Parameters
        ----------
        base : subclass of BaseRepresentation or string
            The type of representation to generate.  Must be a *class*
            (not an instance), or the string name of the representation
            class.
        s : subclass of `~astropy.coordinates.BaseDifferential`, str, optional
            Class in which any velocities should be represented. Must be
            a *class* (not an instance), or the string name of the
            differential class.  If equal to 'base' (default), inferred from
            the base class.  If `None`, all velocity information is dropped.
        in_frame_units : bool, keyword only
            Force the representation units to match the specified units
            particular to this frame

        Returns
        -------
        newrep : BaseRepresentation-derived object
            A new representation object of this frame's `data`.

        Raises
        ------
        AttributeError
            If this object had no `data`

        Examples
        --------
        >>> from astropy import units as u
        >>> from astropy.coordinates import SkyCoord, CartesianRepresentation
        >>> coord = SkyCoord(0*u.deg, 0*u.deg)
        >>> coord.represent_as(CartesianRepresentation)  # doctest: +FLOAT_CMP
        <CartesianRepresentation (x, y, z) [dimensionless]
                (1., 0., 0.)>

        >>> coord.representation_type = CartesianRepresentation
        >>> coord  # doctest: +FLOAT_CMP
        <SkyCoord (ICRS): (x, y, z) [dimensionless]
            (1., 0., 0.)>
        """
        rep = self.frame.represent_as(base, s=s, in_frame_units=in_frame_units)

        return InterpolatedRepresentation(rep, affine=self.affine, **self._interp_kwargs)

    def transform_to(
        self,
        new_frame: Union[BaseCoordinateFrame, SkyCoord],
    ) -> InterpolatedCoordinateFrame:
        """Transform this object's coordinate data to a new frame.

        Parameters
        ----------
        new_frame : frame object or SkyCoord object
            The frame to transform this coordinate frame into.

        Returns
        -------
        transframe
            A new object with the coordinate data represented in the
            ``newframe`` system.

        Raises
        ------
        ValueError
            If there is no possible transformation route.
        """
        newframe = self.frame.transform_to(new_frame)
        return self._realize_class(newframe)

    def copy(self) -> InterpolatedCoordinateFrame:
        interp_kwargs = self._interp_kwargs.copy()
        frame = self.frame.realize_frame(self.data)
        iframe: InterpolatedCoordinateFrame = self._class_(
            frame,
            affine=self.affine.copy(),
            interps=None,
            **interp_kwargs,
        )
        return iframe

    def _frame_attrs_repr(self) -> str:  # FIXME!!
        s: str = self.frame._frame_attrs_repr()
        return s

    def _data_repr(self) -> str:
        """Returns a string representation of the coordinate data.

        This method is modified from the original to include the affine
        parameter.

        Returns
        -------
        str
            string representation of the data
        """
        # if not self.has_data:  # must have data to be interpolated
        #     return ""

        rep_cls = self.representation_type

        if rep_cls:
            if hasattr(rep_cls, "_unit_representation") and isinstance(
                self.frame.data,
                rep_cls._unit_representation,
            ):
                rep_cls = self.frame.data.__class__

            if "s" in self.frame.data.differentials:
                dif_data = self.frame.data.differentials["s"]
                dif_cls = (
                    self.get_representation_cls("s")
                    if not isinstance(dif_data, _UNIT_DIF_TYPES)
                    else dif_data.__class__
                )

            else:
                dif_cls = None

            data = self.represent_as(rep_cls, dif_cls, in_frame_units=True)
            data_repr = repr(data)

            # Generate the list of component names out of the repr string
            part1, _, remainder = data_repr.partition("(")
            if remainder != "":
                comp_str, _, part2 = remainder.partition(")")
                comp_names: List[str] = comp_str.split(", ")

                affine_name, comp_name_0 = comp_names[0].split("| ")
                comp_names[0] = comp_name_0

                # Swap in frame-specific component names
                rep_comp_names = self.representation_component_names
                invnames = {nmrepr: nmpref for nmpref, nmrepr in rep_comp_names.items()}
                for i, name in enumerate(comp_names):
                    comp_names[i] = invnames.get(name, name)

                # Reassemble the repr string
                data_repr = part1 + "(" + affine_name + "| " + ", ".join(comp_names) + ")" + part2

        # else:  # uncomment when encounter
        #     data = self.frame.data
        #     data_repr = repr(self.data)

        data_cls_name = "Interpolated" + data.__class__.__name__
        if data_repr.startswith("<" + data_cls_name):
            # remove both the leading "<" and the space after the name, as well
            # as the trailing ">"
            i = len(data_cls_name) + 2
            data_repr = data_repr[i:-1]
        # else:  # uncomment when encounter
        #     data_repr = "Data:\n" + data_repr

        if "s" in self.data.differentials:
            data_repr_spl = data_repr.split("\n")
            if "has differentials" in data_repr_spl[-1]:
                diffrepr = repr(data.differentials["s"]).split("\n")
                if diffrepr[0].startswith("<"):
                    diffrepr[0] = " " + " ".join(diffrepr[0].split(" ")[1:])
                for frm_nm, rep_nm in self.get_representation_component_names(
                    "s",
                ).items():
                    diffrepr[0] = diffrepr[0].replace(rep_nm, frm_nm)
                if diffrepr[-1].endswith(">"):
                    diffrepr[-1] = diffrepr[-1][:-1]
                data_repr_spl[-1] = "\n".join(diffrepr)

            data_repr = "\n".join(data_repr_spl)

        return data_repr

    def __repr__(self) -> str:
        frameattrs = self._frame_attrs_repr()
        data_repr = self._data_repr()

        if frameattrs:
            frameattrs = f" ({frameattrs})"

        cls_name = self.__class__.__name__
        if data_repr:
            return f"<Interpolated{cls_name} " f"Coordinate{frameattrs}: {data_repr}>"
        # else:  # uncomment when encounter
        #     return f"<Interpolated{cls_name} Frame{frameattrs}>"

        return "TODO!"

    # -----------------------------------------------------
    # Separation

    # @overload
    # def separation(self, point: CoordinateType, interpolate: Literal[True]) -> IUSU:
    #     ...

    # @overload
    # def separation(
    #     self, point: CoordinateType, interpolate: Literal[True], affine: Quantity
    # ) -> IUSU:
    #     ...

    # @overload
    # def separation(self, point: CoordinateType, interpolate: Literal[False]) -> Angle:
    #     ...

    # @overload
    # def separation(
    #     self, point: CoordinateType, interpolate: Literal[False], affine: Quantity
    # ) -> Angle:
    #     ...

    # @overload
    # def separation(
    #     self, point: CoordinateType, interpolate: bool, affine: Optional[Quantity]
    # ) -> Union[Angle, IUSU]:
    #     ...

    def separation(
        self,
        point: CoordinateType,
        *,
        interpolate: bool = True,
        affine: Optional[Quantity] = None,
    ) -> Union[Angle, IUSU]:
        """Computes on-sky separation between this coordinate and another.

        .. note::

            If the ``other`` coordinate object is in a different frame, it is
            first transformed to the frame of this object. This can lead to
            unintuitive behavior if not accounted for. Particularly of note is
            that ``self.separation(other)`` and ``other.separation(self)`` may
            not give the same answer in this case.

        Parameters
        ----------
        other : `~astropy.coordinates.BaseCoordinateFrame`
            The coordinate to get the separation to.
        interpolated : bool, optional keyword-only
            Whether to return the separation as an interpolation, or as points.
        affine : `~astropy.units.Quantity` array-like or None, optional
            The affine interpolation parameter. If None (default), return
            angular width evaluated at all "tick" interpolation points.

        Returns
        -------
        sep : `~astropy.coordinates.Angle` or `~interpolated_coordinates.utils.InterpolatedUnivariateSplinewithUnits`  # noqa: E501
            The on-sky separation between this and the ``other`` coordinate.

        Notes
        -----
        The separation is calculated using the Vincenty formula, which
        is stable at all locations, including poles and antipodes [1]_.

        .. [1] https://en.wikipedia.org/wiki/Great-circle_distance
        """
        return self._separation(point, angular=True, interpolate=interpolate, affine=affine)

    def separation_3d(
        self,
        point: CoordinateType,
        *,
        interpolate: bool = True,
        affine: Optional[Quantity] = None,
    ) -> Union[Distance, IUSU]:
        """
        Computes three dimensional separation between this coordinate
        and another.

        For more on how to use this (and related) functionality, see the
        examples in :doc:`astropy:coordinates/matchsep`.

        Parameters
        ----------
        other : `~astropy.coordinates.BaseCoordinateFrame`
            The coordinate to get the separation to.
        interpolated : bool, optional keyword-only
            Whether to return the separation as an interpolation, or as points.
        affine : `~astropy.units.Quantity` array-like or None, optional
            The affine interpolation parameter. If None (default), return
            angular width evaluated at all "tick" interpolation points.

        Returns
        -------
        sep : `~astropy.coordinates.Distance` or `~interpolated_coordinates.utils.InterpolatedUnivariateSplinewithUnits`  # noqa: E501
            The real-space distance between these two coordinates.

        Raises
        ------
        ValueError
            If this or the other coordinate do not have distances.
        """
        return self._separation(point, angular=False, interpolate=interpolate, affine=affine)

    def _separation(
        self,
        point: SkyCoord,
        angular: bool,
        interpolate: bool,
        affine: Optional[Quantity],
    ) -> Union[Angle, Distance, IUSU]:
        """Separation helper function."""
        affine = self.affine if affine is None else affine

        c = self(affine=affine)
        seps = getattr(c, "separation" if angular else "separation_3d")(point)

        if not interpolate:
            return seps

        return IUSU(affine, seps)  # TODO! if 1 point


#####################################################################


class InterpolatedSkyCoord(SkyCoord):
    """Interpolated SkyCoord."""

    def __init__(
        self,
        *args: Any,
        affine: Optional[Quantity] = None,
        copy: bool = True,
        **kwargs: Any,
    ) -> None:

        keys = tuple(kwargs.keys())  # needed b/c pop changes size
        interp_kwargs = {k: kwargs.pop(k) for k in keys if k.startswith("interp_")}

        super().__init__(*args, copy=copy, **kwargs)

        # change frame to InterpolatedCoordinateFrame
        if not isinstance(self.frame, InterpolatedCoordinateFrame):
            self._sky_coord_frame = InterpolatedCoordinateFrame(
                self.frame, affine=affine, **interp_kwargs
            )

    def __call__(self, affine: Optional[Quantity] = None) -> SkyCoord:
        """Evaluate interpolated representation.

        Parameters
        ----------
        affine : `~astropy.units.Quantity` array-like
            The affine interpolation parameter.
            If None, returns representation points.

        Returns
        -------
        `SkyCoord`
            CoordinateFrame of type ``self.frame`` evaluated with `affine`
        """
        newsc = SkyCoord(self.frame(affine))
        return newsc

    def transform_to(
        self,
        frame: FrameLikeType,
        merge_attributes: bool = True,
    ) -> InterpolatedSkyCoord:
        """Transform this coordinate to a new frame.

        The precise frame transformed to depends on ``merge_attributes``.
        If `False`, the destination frame is used exactly as passed in.
        But this is often not quite what one wants.  E.g., suppose one wants to
        transform an ICRS coordinate that has an obstime attribute to FK4; in
        this case, one likely would want to use this information. Thus, the
        default for ``merge_attributes`` is `True`, in which the precedence is
        as follows: (1) explicitly set (i.e., non-default) values in the
        destination frame; (2) explicitly set values in the source; (3) default
        value in the destination frame.

        Note that in either case, any explicitly set attributes on the source
        `SkyCoord` that are not part of the destination frame's definition are
        kept (stored on the resulting `SkyCoord`), and thus one can round-trip
        (e.g., from FK4 to ICRS to FK4 without loosing obstime).

        Parameters
        ----------
        frame : str, `BaseCoordinateFrame` class or instance, or `SkyCoord` instance
            The frame to transform this coordinate into.  If a `SkyCoord`, the
            underlying frame is extracted, and all other information ignored.
        merge_attributes : bool, optional
            Whether the default attributes in the destination frame are allowed
            to be overridden by explicitly set attributes in the source
            (see note above; default: `True`).

        Returns
        -------
        coord : `SkyCoord`
            A new object with this coordinate represented in the `frame` frame.

        Raises
        ------
        ValueError
            If there is no possible transformation route.
        """
        sc = SkyCoord(self, copy=False)  # TODO, less jank
        nsc = sc.transform_to(frame, merge_attributes=merge_attributes)

        return self.__class__(nsc, affine=self.affine, copy=False)

    # -----------------------------------------------------
    # Separation

    def separation(
        self,
        point: CoordinateType,
        *,
        interpolate: bool = True,
        affine: Optional[Quantity] = None,
    ) -> Union[Angle, IUSU]:
        """Computes on-sky separation between this coordinate and another.

        .. note::

            If the ``other`` coordinate object is in a different frame, it is
            first transformed to the frame of this object. This can lead to
            unintuitive behavior if not accounted for. Particularly of note is
            that ``self.separation(other)`` and ``other.separation(self)`` may
            not give the same answer in this case.

        For more on how to use this (and related) functionality, see the
        examples in :doc:`astropy:coordinates/matchsep`.

        Parameters
        ----------
        other : `~astropy.coordinates.SkyCoord` or `~astropy.coordinates.BaseCoordinateFrame`
            The coordinate to get the separation to.
        interpolated : bool, optional keyword-only
            Whether to return the separation as an interpolation, or as points.
        affine : `~astropy.units.Quantity` array-like or None, optional
            The affine interpolation parameter. If None (default), return
            angular width evaluated at all "tick" interpolation points.

        Returns
        -------
        sep : ~astropy.coordinates.Angle` or `~interpolated_coordinates.utils.InterpolatedUnivariateSplinewithUnits`  # noqa: E501
            The on-sky separation between this and the ``other`` coordinate.

        Notes
        -----
        The separation is calculated using the Vincenty formula, which
        is stable at all locations, including poles and antipodes [1]_.

        .. [1] https://en.wikipedia.org/wiki/Great-circle_distance
        """
        return self._separation(point, angular=True, interpolate=interpolate, affine=affine)

    def separation_3d(
        self,
        point: CoordinateType,
        *,
        interpolate: bool = True,
        affine: Optional[Quantity] = None,
    ) -> Union[Distance, IUSU]:
        """
        Computes three dimensional separation between this coordinate
        and another.

        For more on how to use this (and related) functionality, see the
        examples in :doc:`astropy:coordinates/matchsep`.

        Parameters
        ----------
        other : `~astropy.coordinates.SkyCoord` or `~astropy.coordinates.BaseCoordinateFrame`
            The coordinate to get the separation to.
        interpolated : bool, optional keyword-only
            Whether to return the separation as an interpolation, or as points.
        affine : `~astropy.units.Quantity` array-like or None, optional
            The affine interpolation parameter. If None (default), return
            angular width evaluated at all "tick" interpolation points.

        Returns
        -------
        sep : `~astropy.coordinates.Distance` or `~interpolated_coordinates.utils.InterpolatedUnivariateSplinewithUnits`  # noqa: E501
            The real-space distance between these two coordinates.

        Raises
        ------
        ValueError
            If this or the other coordinate do not have distances.
        """
        return self._separation(point, angular=False, interpolate=interpolate, affine=affine)

    def _separation(
        self,
        point: SkyCoord,
        angular: bool,
        interpolate: bool,
        affine: Optional[Quantity],
    ) -> Union[Angle, Distance, IUSU]:
        """Separation helper function."""
        return InterpolatedCoordinateFrame._separation(
            self,
            point,
            angular=angular,
            interpolate=interpolate,
            affine=affine,
        )

    # ---------------------------------------------------------------

    def match_to_catalog_sky(
        self,
        catalogcoord: CoordinateType,
        nthneighbor: int = 1,
    ) -> Tuple[ndarray, Angle, Quantity]:
        """
        Finds the nearest on-sky matches of this coordinate in a set of
        catalog coordinates.

        For more on how to use this (and related) functionality, see the
        examples in :doc:`astropy:coordinates/matchsep`.

        Parameters
        ----------
        catalogcoord : `~astropy.coordinates.SkyCoord` or `~astropy.coordinates.BaseCoordinateFrame`
            The base catalog in which to search for matches. Typically this
            will be a coordinate object that is an array (i.e.,
            ``catalogcoord.isscalar == False``)
        nthneighbor : int, optional
            Which closest neighbor to search for.  Typically ``1`` is
            desired here, as that is correct for matching one set of
            coordinates to another. The next likely use case is ``2``,
            for matching a coordinate catalog against *itself* (``1``
            is inappropriate because each point will find itself as the
            closest match).

        Returns
        -------
        idx : int array
            Indices into ``catalogcoord`` to get the matched points for
            each of this object's coordinates. Shape matches this
            object.
        sep2d : `~astropy.coordinates.Angle`
            The on-sky separation between the closest match for each
            element in this object in ``catalogcoord``. Shape matches
            this object.
        dist3d : `~astropy.units.Quantity` ['length']
            The 3D distance between the closest match for each element
            in this object in ``catalogcoord``. Shape matches this
            object. Unless both this and ``catalogcoord`` have associated
            distances, this quantity assumes that all sources are at a
            distance of 1 (dimensionless).

        Notes
        -----
        This method requires `SciPy <https://www.scipy.org/>`_ to be
        installed or it will fail.

        See Also
        --------
        astropy.coordinates.match_coordinates_sky
        SkyCoord.match_to_catalog_3d
        """
        idx: ndarray  # pragma: no cover
        sep2d: Angle  # pragma: no cover
        dist3d: Quantity  # pragma: no cover
        idx, sep2d, dist3d = super().match_to_catalog_sky(
            catalogcoord,
            nthneighbor=nthneighbor,
        )  # pragma: no cover
        return idx, sep2d, dist3d  # pragma: no cover

    def match_to_catalog_3d(
        self,
        catalogcoord: CoordinateType,
        nthneighbor: int = 1,
    ) -> Tuple[ndarray, Angle, Quantity]:
        """
        Finds the nearest 3-dimensional matches of this coordinate to a set
        of catalog coordinates.

        This finds the 3-dimensional closest neighbor, which is only different
        from the on-sky distance if ``distance`` is set in this object or the
        ``catalogcoord`` object.

        For more on how to use this (and related) functionality, see the
        examples in :doc:`astropy:coordinates/matchsep`.

        Parameters
        ----------
        catalogcoord : `~astropy.coordinates.SkyCoord` or `~astropy.coordinates.BaseCoordinateFrame`
            The base catalog in which to search for matches. Typically this
            will be a coordinate object that is an array (i.e.,
            ``catalogcoord.isscalar == False``)
        nthneighbor : int, optional
            Which closest neighbor to search for.  Typically ``1`` is
            desired here, as that is correct for matching one set of
            coordinates to another.  The next likely use case is
            ``2``, for matching a coordinate catalog against *itself*
            (``1`` is inappropriate because each point will find
            itself as the closest match).

        Returns
        -------
        idx : int array
            Indices into ``catalogcoord`` to get the matched points for
            each of this object's coordinates. Shape matches this
            object.
        sep2d : `~astropy.coordinates.Angle`
            The on-sky separation between the closest match for each
            element in this object in ``catalogcoord``. Shape matches
            this object.
        dist3d : `~astropy.units.Quantity` ['length']
            The 3D distance between the closest match for each element
            in this object in ``catalogcoord``. Shape matches this
            object.

        Notes
        -----
        This method requires `SciPy <https://www.scipy.org/>`_ to be
        installed or it will fail.

        See Also
        --------
        astropy.coordinates.match_coordinates_3d
        SkyCoord.match_to_catalog_sky
        """
        idx: ndarray  # pragma: no cover
        sep2d: Angle  # pragma: no cover
        dist3d: Quantity  # pragma: no cover
        idx, sep2d, dist3d = super().match_to_catalog_3d(
            catalogcoord,
            nthneighbor=nthneighbor,
        )  # pragma: no cover
        return idx, sep2d, dist3d  # pragma: no cover

    def search_around_sky(
        self,
        searcharoundcoords: CoordinateType,
        seplimit: Quantity,
    ) -> Tuple[ndarray, ndarray, Angle, Quantity]:
        """
        Searches for all coordinates in this object around a supplied set of
        points within a given on-sky separation.

        This is intended for use on `~astropy.coordinates.SkyCoord` objects
        with coordinate arrays, rather than a scalar coordinate.  For a scalar
        coordinate, it is better to use
        `~astropy.coordinates.SkyCoord.separation`.

        For more on how to use this (and related) functionality, see the
        examples in :doc:`astropy:coordinates/matchsep`.

        Parameters
        ----------
        searcharoundcoords : coordinate-like
            The coordinates to search around to try to find matching points in
            this `SkyCoord`. This should be an object with array coordinates,
            not a scalar coordinate object.
        seplimit : `~astropy.units.Quantity` ['angle']
            The on-sky separation to search within.

        Returns
        -------
        idxsearcharound : int array
            Indices into ``searcharoundcoords`` that match the
            corresponding elements of ``idxself``. Shape matches
            ``idxself``.
        idxself : int array
            Indices into ``self`` that match the
            corresponding elements of ``idxsearcharound``. Shape matches
            ``idxsearcharound``.
        sep2d : `~astropy.coordinates.Angle`
            The on-sky separation between the coordinates. Shape matches
            ``idxsearcharound`` and ``idxself``.
        dist3d : `~astropy.units.Quantity` ['length']
            The 3D distance between the coordinates. Shape matches
            ``idxsearcharound`` and ``idxself``.

        Notes
        -----
        This method requires `SciPy <https://www.scipy.org/>`_ to be
        installed or it will fail.

        In the current implementation, the return values are always sorted in
        the same order as the ``searcharoundcoords`` (so ``idxsearcharound`` is
        in ascending order).  This is considered an implementation detail,
        though, so it could change in a future release.

        See Also
        --------
        astropy.coordinates.search_around_sky
        SkyCoord.search_around_3d
        """
        idxsearch: ndarray  # pragma: no cover
        idxself: ndarray  # pragma: no cover
        sep2d: Angle  # pragma: no cover
        dist3d: Quantity  # pragma: no cover
        idxsearch, idxself, sep2d, dist3d = super().search_around_sky(
            searcharoundcoords,
            seplimit,
        )  # pragma: no cover
        return idxsearch, idxself, sep2d, dist3d  # pragma: no cover

    def search_around_3d(
        self,
        searcharoundcoords: CoordinateType,
        distlimit: Quantity,
    ) -> Tuple[ndarray, ndarray, Angle, Quantity]:
        """
        Searches for all coordinates in this object around a supplied set of
        points within a given 3D radius.

        This is intended for use on `~astropy.coordinates.SkyCoord` objects
        with coordinate arrays, rather than a scalar coordinate.  For a scalar
        coordinate, it is better to use
        `~astropy.coordinates.SkyCoord.separation_3d`.

        For more on how to use this (and related) functionality, see the
        examples in :doc:`astropy:coordinates/matchsep`.

        Parameters
        ----------
        searcharoundcoords : SkyCoord or `~astropy.coordinates.BaseCoordinateFrame`
            The coordinates to search around to try to find matching points in
            this `SkyCoord`. This should be an object with array coordinates,
            not a scalar coordinate object.
        distlimit : `~astropy.units.Quantity` ['length']
            The physical radius to search within.

        Returns
        -------
        idxsearcharound : int array
            Indices into ``searcharoundcoords`` that match the
            corresponding elements of ``idxself``. Shape matches
            ``idxself``.
        idxself : int array
            Indices into ``self`` that match the
            corresponding elements of ``idxsearcharound``. Shape matches
            ``idxsearcharound``.
        sep2d : `~astropy.coordinates.Angle`
            The on-sky separation between the coordinates. Shape matches
            ``idxsearcharound`` and ``idxself``.
        dist3d : `~astropy.units.Quantity` ['length']
            The 3D distance between the coordinates. Shape matches
            ``idxsearcharound`` and ``idxself``.

        Notes
        -----
        This method requires `SciPy <https://www.scipy.org/>`_ to be
        installed or it will fail.

        In the current implementation, the return values are always sorted in
        the same order as the ``searcharoundcoords`` (so ``idxsearcharound`` is
        in ascending order).  This is considered an implementation detail,
        though, so it could change in a future release.

        See Also
        --------
        astropy.coordinates.search_around_3d
        SkyCoord.search_around_sky
        """
        idxsearch: ndarray  # pragma: no cover
        idxself: ndarray  # pragma: no cover
        sep2d: Angle  # pragma: no cover
        dist3d: Quantity  # pragma: no cover
        idxsearch, idxself, sep2d, dist3d = super().search_around_3d(
            searcharoundcoords,
            distlimit,
        )  # pragma: no cover
        return idxsearch, idxself, sep2d, dist3d  # pragma: no cover
