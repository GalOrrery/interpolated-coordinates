# -*- coding: utf-8 -*-

"""Testing :mod:`~interpolated_coordinates.frame`."""

__all__ = [
    "Test_InterpolatedCoordinateFrame",
    "Test_InterpolatedSkyCoord",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest

# LOCAL
from .test_representation import InterpolatedCoordinatesBase
from interpolated_coordinates.frame import (
    InterpolatedCoordinateFrame,
    InterpolatedRepresentation,
    InterpolatedSkyCoord,
)
from interpolated_coordinates.utils import InterpolatedUnivariateSplinewithUnits

##############################################################################
# TESTS
##############################################################################


class Test_InterpolatedCoordinateFrame(InterpolatedCoordinatesBase):
    """Test :class:`~{package}.{klass}`."""

    @pytest.fixture(scope="class")
    def irep(self, rep, affine):
        return InterpolatedRepresentation(rep, affine=affine)

    @pytest.fixture(scope="class")
    def frame(self):
        return coord.Galactocentric

    @pytest.fixture(scope="class")
    def crd(self, rep, frame):
        return frame(rep)

    @pytest.fixture(scope="class")
    def icrd_cls(self):
        return InterpolatedCoordinateFrame

    @pytest.fixture(scope="class")
    def icrd(self, icrd_cls, frame, irep):
        return icrd_cls(frame(irep))

    #######################################################
    # Method Tests

    def test_init(self, icrd_cls, frame, irep, rep, affine) -> None:
        """Test method ``__init__``."""
        # -------------------
        # rep is interpolated

        c = icrd_cls(frame(irep))

        assert isinstance(c, icrd_cls)
        assert isinstance(c.data, InterpolatedRepresentation)

        # -------------------
        # rep is base astropy

        # doesn't work b/c no affine
        with pytest.raises(ValueError, match="`data` is not already interpolated"):
            icrd_cls(frame(rep), affine=None)

        # ----

        # works with affine
        c = icrd_cls(frame(rep), affine=affine)

        assert isinstance(c, icrd_cls)
        assert isinstance(c.data, InterpolatedRepresentation)

        # -------------------
        # rep is wrong type

        class Obj:
            data = object()

        with pytest.raises(TypeError, match="`data` must be type "):
            icrd_cls(Obj())

    def test__interp_kwargs(self, icrd) -> None:
        """Test method ``_interp_kwargs``."""
        # property get
        assert icrd._interp_kwargs is icrd.data._interp_kwargs

        # setter
        icrd._interp_kwargs = {"a": 1}
        assert icrd.data._interp_kwargs["a"] == 1
        icrd._interp_kwargs = {}  # reset

    def test_call(self, icrd, frame, num) -> None:
        """Test method ``__call__``.

        Since it returns a BaseCoordinateFrame, and does the evaluation
        through the InterpolatedRepresentation, all we need to test here
        is that it's the right type.

        """
        data = icrd()

        assert isinstance(data, frame)
        assert len(data) == num

    def test__class_(self, icrd_cls, icrd) -> None:
        """Test method ``_class_``."""
        assert issubclass(icrd._class_, icrd_cls)

    @pytest.mark.skip("TODO")
    def test__realize_class(self, icrd) -> None:
        """Test method ``_realize_class``."""
        assert False

    @pytest.mark.skip("TODO")
    def test_realize_frame(self, icrd) -> None:
        """Test method ``realize_frame``."""
        assert False

    def test_derivative(self, icrd, affine) -> None:
        """Test method ``derivative``.

        Just passes to the Representation.

        """
        # --------------------

        ideriv = icrd.derivative(n=1)  # a straight line

        assert all(ideriv.affine == affine)
        assert np.allclose(ideriv._values.view(float), 0.1)

        # --------------------

        ideriv = icrd.derivative(n=2)  # no 2nd deriv

        assert all(ideriv.affine == affine)
        assert np.allclose(ideriv._values.view(float), 0.0)

        # --------------------

        ideriv = icrd.derivative(n=3)  # no 3rd deriv

        assert all(ideriv.affine == affine)
        assert np.allclose(ideriv._values.view(float), 0.0)

    def test_affine(self, icrd, affine) -> None:
        """Test method ``affine``.

        Just passes to the Representation.

        """
        assert all(icrd.affine == affine)
        assert all(icrd.frame.data.affine == affine)

    def test_headless_tangent_vectors(self, icrd_cls, icrd) -> None:
        """Test method ``headless_tangent_vectors``.

        Wraps Representation in InterpolatedCoordinateFrame

        """
        htv = icrd.headless_tangent_vectors()

        assert isinstance(htv, icrd_cls)  # interp
        assert isinstance(htv, coord.BaseCoordinateFrame)

        for c in htv.data.components:
            assert np.allclose(getattr(htv.data, c), 0.1 * u.kpc)

    def test_tangent_vectors(self, icrd_cls, icrd) -> None:
        """Test method ``tangent_vectors``.

        Wraps Representation in InterpolatedCoordinateFrame

        """
        tv = icrd.tangent_vectors()

        assert isinstance(tv, icrd_cls)  # interp
        assert isinstance(tv, coord.BaseCoordinateFrame)

        for c in tv.data.components:
            assert np.allclose(
                getattr(tv.data, c) - getattr(icrd.data, c),
                0.1 * u.kpc,
            )

    def test___class__(self, icrd_cls, icrd) -> None:
        """Test method ``__class__``.

        Just passes to the CoordinateFrame.
        """
        assert icrd.__class__ is icrd.frame.__class__
        assert issubclass(icrd.__class__, coord.BaseCoordinateFrame)
        assert isinstance(icrd, icrd_cls)

        # Cannot set the class
        with pytest.raises(TypeError):
            icrd.__class__ = coord.BaseCoordinateFrame

    def test___getattr__(self, icrd, num) -> None:
        """Test method ``__getattr__``.

        Routes everything to underlying CoordinateFrame.
        Lets just test the ``shape``.

        """
        assert icrd.shape == icrd.frame.shape
        assert icrd.shape == (num,)

        assert icrd.ndim == icrd.frame.ndim
        assert icrd.ndim == 1

    def test_len(self, icrd, num) -> None:
        """Test method ``__len__``."""
        assert len(icrd) == num

    def test___getitem__(self, icrd_cls, icrd) -> None:
        """Test method ``__getitem__``."""
        # Test has problem when slicing with <3 elements
        # TODO? fix?
        with pytest.raises(Exception):
            icrd[:3]

        # works otherwise
        inst = icrd[:4]

        assert isinstance(inst, coord.BaseCoordinateFrame)
        assert isinstance(inst, icrd_cls)
        assert isinstance(inst, icrd.__class__)
        assert isinstance(inst, icrd._class_)

        assert inst.representation_type == icrd.representation_type

        assert len(inst) == 4

    def test_transform_to(self, icrd) -> None:
        """Test method ``transform_to``.

        All the transformation is handled in the frame. Only need to
        test that it's still interpolated.

        """
        newinst = icrd.transform_to(coord.HeliocentricTrueEcliptic())

        assert isinstance(newinst, coord.HeliocentricTrueEcliptic)
        assert isinstance(newinst, InterpolatedCoordinateFrame)

        assert isinstance(newinst.frame, coord.HeliocentricTrueEcliptic)

        assert isinstance(newinst.frame.data, coord.CartesianRepresentation)
        assert isinstance(newinst.frame.data, InterpolatedRepresentation)

        assert isinstance(newinst.frame.data.data, coord.CartesianRepresentation)

    def test_copy(self, icrd) -> None:
        """Test method ``copy``."""
        newrep = icrd.copy()

        assert newrep is not icrd  # not the same object
        assert isinstance(newrep, InterpolatedCoordinateFrame)

        # TODO more tests

    def test__frame_attrs_repr(self, icrd) -> None:
        """Test method ``_frame_attrs_repr``."""
        assert icrd._frame_attrs_repr() == icrd.frame._frame_attrs_repr()
        # TODO more tests

    def test__data_repr(self, icrd) -> None:
        """Test method ``_data_repr``."""
        data_repr = icrd._data_repr()
        assert isinstance(data_repr, str)
        # TODO more tests

    def test___repr__(self, icrd) -> None:
        """Test method ``__repr__``."""
        s = icrd.__repr__()
        assert isinstance(s, str)

        # a test for unit dif types
        icrd.representation_type = coord.UnitSphericalRepresentation
        s = icrd.__repr__()
        assert isinstance(s, str)

        # TODO more tests

    def test_separation(self, icrd, crd, affine):
        """Test method ``separation``."""
        assert all(crd.separation(crd) == 0)  # null hypothesis

        # Interpolated coordinate separation is similar
        assert np.allclose(icrd.separation(crd, interpolate=False), 0)
        assert np.allclose(crd.separation(icrd), 0)

        # Can also return an interpolation
        separation = icrd.separation(crd, interpolate=True)
        assert isinstance(separation, InterpolatedUnivariateSplinewithUnits)
        assert np.allclose(separation(affine), 0)

    def test_separation_3d(self, icrd, crd, affine):
        """Test method ``separation_3d``."""
        assert all(crd.separation_3d(crd) == 0)  # null hypothesis

        # Interpolated coordinate separation is similar
        assert np.allclose(icrd.separation_3d(crd, interpolate=False), 0)
        assert np.allclose(crd.separation_3d(icrd), 0)

        # Can also return an interpolation
        separation_3d = icrd.separation_3d(crd, interpolate=True)
        assert isinstance(separation_3d, InterpolatedUnivariateSplinewithUnits)
        assert np.allclose(separation_3d(affine), 0)


#####################################################################


class Test_InterpolatedSkyCoord(InterpolatedCoordinatesBase):
    """Test InterpolatedSkyCoord."""

    @pytest.fixture(scope="class")
    def irep(self, rep, affine):
        return InterpolatedRepresentation(rep, affine=affine)

    @pytest.fixture(scope="class")
    def frame(self):
        return coord.Galactocentric

    @pytest.fixture(scope="class")
    def crd(self, frame, rep):
        return frame(rep)

    @pytest.fixture(scope="class")
    def icrd_cls(self):
        return InterpolatedCoordinateFrame

    @pytest.fixture(scope="class")
    def icrd(self, icrd_cls, frame, irep):
        return icrd_cls(frame(irep))

    @pytest.fixture(scope="class")
    def scrd(self, crd):
        return coord.SkyCoord(crd)

    @pytest.fixture(scope="class")
    def iscrd_cls(self):
        return InterpolatedSkyCoord

    @pytest.fixture(scope="class")
    def iscrd(self, iscrd_cls, icrd):
        return iscrd_cls(icrd)

    #######################################################
    # Method Tests

    def _test_isc(
        self,
        isc,
        representation_type=coord.UnitSphericalRepresentation,
    ) -> None:
        """Runs through all the levels, testing type."""
        inst = isc.transform_to(coord.FK5())

        assert isinstance(inst, coord.SkyCoord)
        assert isinstance(inst, InterpolatedSkyCoord)

        assert isinstance(inst.frame, coord.FK5)

        assert isinstance(inst.frame.data, representation_type)
        assert isinstance(inst.frame.data, InterpolatedRepresentation)

        assert isinstance(inst.frame.data.data, representation_type)

    def test_init(self, iscrd_cls, num, affine) -> None:
        """Test method ``__init__``.

        Copying from astropy docs

        """
        # -----------
        c = iscrd_cls(
            [10] * num,
            [20] * num,
            unit="deg",
            affine=affine,
        )  # defaults to ICRS frame
        self._test_isc(c)

        # -----------
        c = iscrd_cls(
            [1, 2, 3, 4],
            [-30, 45, 8, 16],
            frame="icrs",
            unit="deg",
            affine=affine[:4],
        )  # 4 coords
        self._test_isc(c)

        # -----------
        coords = [
            "1:12:43.2 +31:12:43",
            "1:12:43.2 +31:12:43",
            "1:12:43.2 +31:12:43",
            "1 12 43.2 +31 12 43",
        ]
        c = iscrd_cls(
            coords,
            frame=coord.FK4,
            unit=(u.hourangle, u.deg),
            obstime="J1992.21",
            affine=affine[:4],
        )
        self._test_isc(c)

        # -----------
        c = iscrd_cls(
            ["1h12m43.2s +1d12m43s"] * num,
            frame=coord.Galactic,
            affine=affine,
        )  # Units from string
        self._test_isc(c)

        # # -----------
        c = iscrd_cls(
            frame="galactic",
            l=["1h12m43.2s"] * num,
            b="+1d12m43s",  # NOT BROADCASTING THIS ONE
            affine=affine,
        )
        self._test_isc(c)

        # -----------
        ra = coord.Longitude([1, 2, 3, 4], unit=u.deg)  # Could also use Angle
        dec = np.array([4.5, 5.2, 6.3, 7.4]) * u.deg  # Astropy Quantity
        c = iscrd_cls(
            ra,
            dec,
            frame="icrs",
            affine=affine[:4],
        )
        self._test_isc(c)

        # -----------
        c = iscrd_cls(
            frame=coord.ICRS,
            ra=ra,
            dec=dec,
            obstime="2001-01-02T12:34:56",
            affine=affine[:4],
        )
        self._test_isc(c)

        # -----------
        c = coord.FK4(
            [1] * num * u.deg,
            2 * u.deg,
        )  # Uses defaults for obstime, equinox
        c = iscrd_cls(
            c,
            obstime="J2010.11",
            equinox="B1965",
            affine=affine,
        )  # Override defaults
        self._test_isc(c)

        # -----------
        c = iscrd_cls(
            w=[0] * num,
            u=1,
            v=2,
            unit="kpc",
            frame="galactic",
            representation_type="cartesian",
            affine=affine,
        )
        self._test_isc(c, representation_type=coord.CartesianRepresentation)

        # -----------
        c = iscrd_cls(
            [
                coord.ICRS(ra=1 * u.deg, dec=2 * u.deg),
                coord.ICRS(ra=3 * u.deg, dec=4 * u.deg),
            ]
            * (num // 2),
            affine=affine,
        )
        self._test_isc(c)

    def test_call(self, iscrd, num) -> None:
        """Test method ``__call__``."""
        inst = iscrd()

        assert isinstance(inst, coord.SkyCoord)
        assert len(inst) == num

    def test_transform_to(self, iscrd_cls, iscrd, affine) -> None:
        """Test method ``transform_to``."""
        for frame in (coord.ICRS,):
            inst = iscrd.transform_to(frame())

            assert isinstance(inst, coord.SkyCoord)
            assert isinstance(inst, iscrd_cls)

            assert isinstance(inst.frame, frame)

            assert all(inst.affine == affine)

    def test_separation(self, iscrd, selfcrd, affine):
        """Test method ``separation``."""
        assert all(scrd.separation(scrd) == 0)  # null hypothesis

        # Interpolated coordinate separation is similar
        assert np.allclose(iscrd.separation(scrd, interpolate=False), 0)
        assert np.allclose(scrd.separation(iscrd), 0)

        # Can also return an interpolation
        separation = iscrd.separation(scrd, interpolate=True)
        assert isinstance(separation, InterpolatedUnivariateSplinewithUnits)
        assert np.allclose(separation(affine), 0)

    def test_separation_3d(self, iscrd, scrd, affine):
        """Test method ``separation_3d``."""
        assert all(scrd.separation_3d(scrd) == 0)  # null hypothesis

        # Interpolated coordinate separation is similar
        assert np.allclose(iscrd.separation_3d(scrd, interpolate=False), 0)
        assert np.allclose(scrd.separation_3d(iscrd), 0)

        # Can also return an interpolation
        separation_3d = iscrd.separation_3d(scrd, interpolate=True)
        assert isinstance(separation_3d, InterpolatedUnivariateSplinewithUnits)
        assert np.allclose(separation_3d(affine), 0)

    def test_match_to_catalog_sky(self) -> None:
        """Test method ``match_to_catalog_sky``."""
        pass  # it just calls super b/c docstring issues

    def test_match_to_catalog_3d(self) -> None:
        """Test method ``match_to_catalog_3d``."""
        pass  # it just calls super b/c docstring issues

    def test_search_around_sky(self) -> None:
        """Test method ``search_around_sky``."""
        pass  # it just calls super b/c docstring issues

    def test_search_around_3d(self) -> None:
        """Test method ``search_around_3d``."""
        pass  # it just calls super b/c docstring issues


#####################################################################
# Tests for embedding an InterpolatedX in something


@pytest.mark.skip("TODO")
def test_InterpolatedRepresentation_in_CoordinateFrame():
    assert False


@pytest.mark.skip("TODO")
def test_InterpolatedCoordinateFrame_in_SkyCoord():
    assert False


@pytest.mark.skip("TODO")
def test_InterpolatedRepresentation_in_CoordinateFrame_in_SkyCoord():
    assert False
