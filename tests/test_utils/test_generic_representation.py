"""Testing :mod:`~interpolated_coordinates.utils.generic_representation`."""

__all__ = [
    "test_GENERIC_REGISTRY",
    "Test_GenericRepresentation",
]


##############################################################################
# IMPORTS

import astropy.coordinates as coord
import astropy.units as u
import pytest

from interpolated_coordinates.utils import generic_representation as gr

##############################################################################
# TESTS
##############################################################################


def test_getattr():
    # failed get
    with pytest.raises(AttributeError):
        gr.this_is_not_an_attribute  # noqa: B018

    # get a representation
    assert issubclass(gr.GenericRadialRepresentation, coord.RadialRepresentation)
    assert issubclass(gr.GenericRadialRepresentation, gr.GenericRepresentation)

    # get a differential
    assert issubclass(gr.GenericRadialDifferential, coord.RadialDifferential)
    assert issubclass(gr.GenericRadialDifferential, gr.GenericDifferential)

    assert issubclass(gr.GenericRadial2ndDifferential, coord.RadialDifferential)
    assert issubclass(gr.GenericRadial2ndDifferential, gr.GenericDifferential)


def test_GENERIC_REGISTRY():  # noqa: N802
    """Test :obj:`~interpolated_coordinates.utils.generic_representation._GENERIC_REGISTRY`."""
    # Check type
    assert isinstance(gr._GENERIC_REGISTRY, dict)

    # Check entries
    for key, val in gr._GENERIC_REGISTRY.items():
        if not isinstance(key, str):
            assert issubclass(key, coord.BaseRepresentationOrDifferential)

        assert issubclass(
            val,
            (gr.GenericRepresentation, gr.GenericDifferential),
        )


#####################################################################


class Test_GenericRepresentation:
    """Test `interpolated_coordinates.utils.generic_representation.GenericRepresentation`."""

    @pytest.fixture(scope="class")
    def rep_cls(self):
        return gr.GenericRepresentation

    # ===============================================================

    def test_attr_classes(self, rep_cls):
        """Test attribute ``attr_classes``."""
        # works as class attribute
        assert tuple(rep_cls.attr_classes.keys()) == ("q1", "q2", "q3")

    def test_init(self, rep_cls):
        """It's abstract, so can't init."""
        with pytest.raises(TypeError, match="Can't instantiate abstract"):
            rep_cls(**{k: i for i, k in enumerate(rep_cls.attr_classes.keys())})

    def test_make_generic_cls(self, rep_cls):
        """Test function ``_make_generic_cls``."""
        # ------------------
        # Already generic

        got = rep_cls._make_generic_cls(gr.GenericRepresentation)
        assert got is gr.GenericRepresentation  # pass thru unchanged

        # ------------------
        # Need to make (or cached)

        got = rep_cls._make_generic_cls(coord.CartesianRepresentation)

        assert gr._GENERIC_REGISTRY  # not empty anymore
        assert got is gr.GenericCartesianRepresentation  # cached

        # ------------------
        # Definitely cached

        expected = got
        got = rep_cls._make_generic_cls(coord.CartesianRepresentation)

        assert got is expected
        assert got is gr.GenericCartesianRepresentation  # cached


class Test_GenericRepresentationSubclass(Test_GenericRepresentation):
    @pytest.fixture(scope="class", autouse=True)
    def _setup_cleanup(self, rep_cls):
        yield  # run tests

        coord.representation.REPRESENTATION_CLASSES.pop("genericrepresentationsubclass")

    @pytest.fixture(scope="class")
    def rep_cls(self):
        class GenericRepresentationSubClass(gr.GenericRepresentation):
            attr_classes = {"q1": u.Quantity, "q2": u.Quantity, "q3": u.Quantity}

            def from_cartesian(self):
                pass

            def to_cartesian(self):
                pass

        return GenericRepresentationSubClass

    @pytest.fixture(scope="class")
    def rep(self, rep_cls):
        return rep_cls(q1=1, q2=2, q3=3)

    # ===============================================================

    def test_attr_classes(self, rep_cls, rep):
        """Test attribute ``attr_classes``."""
        super().test_attr_classes(rep_cls)

        # works as instance attribute
        assert tuple(rep.attr_classes.keys()) == ("q1", "q2", "q3")

    @pytest.mark.parametrize(
        ("q1", "q2", "q3"),
        [
            (1, 2, 3),  # no units
            (1 * u.km, 2, 3),  # mixed units
            (1, 2, 3) * u.deg,  # all units
        ],
    )
    def test_init(self, rep_cls, q1, q2, q3):
        rep = rep_cls(q1=q1, q2=q2, q3=q3)
        assert (rep.q1, rep.q2, rep.q3) == (q1, q2, q3)


class Test_GenericCartesianRepresentation(Test_GenericRepresentation):
    @pytest.fixture(scope="class")
    def rep_cls(self):
        return gr.GenericRepresentation._make_generic_cls(coord.CartesianRepresentation)

    @pytest.fixture(scope="class")
    def rep(self, rep_cls):
        return rep_cls(x=1, y=2, z=3)

    # ===============================================================

    def test_attr_classes(self, rep_cls, rep):
        """Test attribute ``attr_classes``."""
        # works as class attribute
        assert tuple(rep_cls.attr_classes.keys()) == ("x", "y", "z")

        # works as instance attribute
        assert tuple(rep.attr_classes.keys()) == ("x", "y", "z")

    def test_init(self, rep_cls):
        inst = rep_cls(x=1, y=2, z=3)

        assert inst.x == 1
        assert inst.y == 2
        assert inst.z == 3


#####################################################################


class Test_GenericDifferential:
    """Test `interpolated_coordinates.utils.generic_representation.GenericDifferential`."""

    @pytest.fixture(scope="class")
    def rep_cls(self):
        return gr.GenericRepresentation

    @pytest.fixture(scope="class")
    def dif_cls(self):
        return gr.GenericDifferential

    @pytest.fixture(scope="class")
    def dif(self, dif_cls):
        return dif_cls(d_q1=1, d_q2=2, d_q3=3)

    # ===============================================================

    def test_attr_classes(self, dif_cls):
        """Test attribute ``attr_classes``."""
        # works as class attribute
        assert tuple(dif_cls.attr_classes.keys()) == ("d_q1", "d_q2", "d_q3")

    def test_base_representation(self, dif_cls, dif):
        """Test attribute ``attr_classes``."""
        # works as class attribute
        assert dif_cls.base_representation == gr.GenericRepresentation

        # works as instance attribute
        assert dif.base_representation == gr.GenericRepresentation

    def test_init(self, rep_cls):
        """It's abstract, so can't init."""
        with pytest.raises(TypeError, match="Can't instantiate abstract"):
            rep_cls(**{k: i for i, k in enumerate(rep_cls.attr_classes.keys())})

    # ---------------------------------------------------------------

    def test_make_generic_cls(self, dif_cls):
        """Test function ``_make_generic_cls``."""
        # ------------------
        # already generic

        got = dif_cls._make_generic_cls(gr.GenericDifferential)
        assert got is gr.GenericDifferential  # pass thru unchangeds

        # ------------------
        # n too small

        with pytest.raises(ValueError, match="n < 1"):
            dif_cls._make_generic_cls(coord.SphericalDifferential, n=0)

        # ------------------
        # need to make, n=1

        got = dif_cls._make_generic_cls(coord.SphericalDifferential, n=1)
        assert got is gr.GenericSphericalDifferential  # cached

        # ------------------
        # need to make, n=2

        got = dif_cls._make_generic_cls(coord.SphericalDifferential, n=2)
        assert got is gr.GenericSpherical2ndDifferential  # cached

        # ------------------
        # cached

        got = dif_cls._make_generic_cls(coord.SphericalDifferential, n=1)
        assert got is gr.GenericSphericalDifferential  # cached

    def test_make_generic_cls_for_representation(self, dif_cls):
        """Test function ``_make_generic_cls_for_representation``."""
        # ------------------
        # n=1, and not in generics' registry

        got = dif_cls._make_generic_cls_for_representation(
            coord.PhysicsSphericalRepresentation,
            n=1,
        )
        assert got is gr.GenericPhysicsSphericalDifferential

        # ------------------
        # do again, getting from registry

        got = dif_cls._make_generic_cls_for_representation(
            coord.PhysicsSphericalRepresentation,
            n=1,
        )
        assert got is gr.GenericPhysicsSphericalDifferential

        # ------------------
        # n=2, and not in generics' registry

        got = dif_cls._make_generic_cls_for_representation(
            coord.PhysicsSphericalRepresentation,
            n=2,
        )
        assert got is gr.GenericPhysicsSpherical2ndDifferential

        # ------------------
        # do again, getting from registry

        got = dif_cls._make_generic_cls_for_representation(
            coord.PhysicsSphericalRepresentation,
            n=2,
        )
        assert got is gr.GenericPhysicsSpherical2ndDifferential


class Test_GenericDifferentialSubClass:
    @pytest.fixture(scope="class", autouse=True)
    def _setup_cleanup(self, rep_cls, dif_cls):
        yield  # run tests

        coord.representation.REPRESENTATION_CLASSES.pop("genericrepresentationsubclass")
        coord.representation.DIFFERENTIAL_CLASSES.pop("genericdifferentialsubclass")

    @pytest.fixture(scope="class")
    def rep_cls(self):
        class GenericRepresentationSubClass(gr.GenericRepresentation):
            attr_classes = {"q1": u.Quantity, "q2": u.Quantity, "q3": u.Quantity}

            def from_cartesian(self):
                pass

            def to_cartesian(self):
                pass

        return GenericRepresentationSubClass

    @pytest.fixture(scope="class")
    def dif_cls(self, rep_cls):
        class GenericDifferentialSubClass(gr.GenericDifferential):
            base_representation = rep_cls

            def from_cartesian(self):
                pass

            def to_cartesian(self):
                pass

        return GenericDifferentialSubClass

    @pytest.fixture(scope="class")
    def dif(self, dif_cls):
        return dif_cls(d_q1=1, d_q2=2, d_q3=3)

    # ===============================================================

    @pytest.mark.parametrize(
        ("q1", "q2", "q3"),
        [
            (1, 2, 3),  # no units
            (1 * u.km, 2, 3),  # mixed units
            (1, 2, 3) * u.deg,  # all units
        ],
    )
    def test_init(self, rep_cls, q1, q2, q3):
        rep = rep_cls(q1=q1, q2=q2, q3=q3)
        assert (rep.q1, rep.q2, rep.q3) == (q1, q2, q3)


class TestGenericCartesianDifferential:
    @pytest.fixture(scope="class")
    def rep_cls(self):
        return coord.CartesianDifferential

    @pytest.fixture(scope="class")
    def dif_cls(self, rep_cls):
        return gr.GenericDifferential._make_generic_cls(rep_cls)

    @pytest.fixture(scope="class")
    def dif(self, dif_cls):
        return dif_cls(d_x=1, d_y=2, d_z=3)


class TestGenericCylindrical3rdDifferential:
    @pytest.fixture(scope="class")
    def rep_cls(self):
        return coord.CylindricalDifferential

    @pytest.fixture(scope="class")
    def dif_cls(self, rep_cls):
        return gr.GenericDifferential._make_generic_cls(rep_cls)

    @pytest.fixture(scope="class")
    def dif(self, dif_cls):
        return dif_cls(d_rho=1, d_phi=2, d_z=3)
