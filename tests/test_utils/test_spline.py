"""Testing :mod:`~interpolated_coordinates.utils.splines`."""

__all__ = [
    "Test_UnivariateSplinewithUnits",
    "Test_InterpolatedUnivariateSplinewithUnits",
    "Test_LSQUnivariateSplinewithUnits",
]

import astropy.units as u
import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose

from interpolated_coordinates.utils import spline

##############################################################################
# TESTS
##############################################################################


class Test_UnivariateSplinewithUnits:
    """Test UnivariateSplinewithUnits."""

    @pytest.fixture(scope="class")
    def num(self):
        return 40

    @pytest.fixture(scope="class")
    def x(self, num):
        return np.linspace(0, 180, num=num) * u.deg

    @pytest.fixture(scope="class")
    def y(self, num):
        return np.linspace(0, 10, num=num) * u.m

    @pytest.fixture(scope="class")
    def w(self, num):
        return np.random.default_rng().uniform(0, 1, size=num)

    @pytest.fixture(scope="class")
    def extra_args(self):
        return {"s": None, "k": 3, "ext": 0, "check_finite": False}

    @pytest.fixture(scope="class")
    def bbox(self):
        return [0 * u.deg, 180 * u.deg]

    @pytest.fixture(scope="class")
    def ispline_cls(self):
        return spline.UnivariateSplinewithUnits

    @pytest.fixture(
        scope="class",
        params=(
            {"w": None, "bbox": [None, None]},
            {"w": ..., "bbox": [None, None]},
            {"w": None, "bbox": ...},
        ),
    )
    def spls(self, request, ispline_cls, x, y, w, bbox, extra_args):
        _w, _bbox = request.param.values()
        w = w if _w is ... else _w
        bbox = bbox if _bbox is ... else _bbox

        return ispline_cls(x, y, w=w, bbox=bbox, **extra_args)

    # -------------------------------

    def test_fail_init(self, ispline_cls, x, y):
        """Test a failed initialization b/c wrong units."""
        bad_unit = x.unit / u.m
        with pytest.raises(u.UnitConversionError):
            ispline_cls(x, y, bbox=[0 * bad_unit, 180 * bad_unit])

    def test_x_unit(self, spls):
        assert spls.x_unit is spls._xunit

    def test_y_unit(self, spls):
        assert spls.y_unit is spls._yunit

    def test_call(self, spls, x, y):
        """Test call method."""
        got = spls(x)  # evaluate spline
        assert_quantity_allclose(got, y, atol=1e-13 * y.unit)

    def test_get_knots(self, spls, x):
        """Test method ``get_knots``."""
        knots = spls.get_knots()
        assert knots.unit == x.unit

    def test_get_coeffs(self, spls, y):
        """Test method ``get_coeffs``."""
        coeffs = spls.get_coeffs()
        assert coeffs.unit == y.unit

    def test_get_residual(self, spls, y):
        """Test method ``get_residual``."""
        residual = spls.get_residual()
        assert residual.unit == y.unit

    def test_integral(self, spls, x, y):
        """Test method ``integral``."""
        integral = spls.integral(x[0], x[-1])
        assert integral.unit == x.unit * y.unit

    def test_derivatives(self, spls, x, y):
        """Test method ``derivatives``."""
        derivatives = spls.derivatives(x[3])
        assert derivatives[0].unit == y.unit

    def test_roots(self, spls, x):
        """Test method ``roots``."""
        roots = spls.roots()
        assert roots.unit == x.unit

    def test_derivative(self, spls, x, y):
        """Test method ``derivative``."""
        deriv = spls.derivative(n=2)
        assert deriv._xunit == x.unit
        assert deriv._yunit == y.unit / x.unit**2

    def test_antiderivative(self, spls, x, y):
        """Test method ``antiderivative``."""
        antideriv = spls.antiderivative(n=2)
        assert antideriv._xunit == x.unit
        assert antideriv._yunit == y.unit * x.unit**2


# -------------------------------------------------------------------


class Test_InterpolatedUnivariateSplinewithUnits(Test_UnivariateSplinewithUnits):
    """Test UnivariateSplinewithUnits."""

    @pytest.fixture(scope="class")
    def extra_args(self):
        return {"k": 3, "ext": 0, "check_finite": False}

    @pytest.fixture(scope="class")
    def ispline_cls(self):
        return spline.InterpolatedUnivariateSplinewithUnits


# -------------------------------------------------------------------


class Test_LSQUnivariateSplinewithUnits(Test_UnivariateSplinewithUnits):
    """Test LSQUnivariateSplinewithUnits."""

    @pytest.fixture(scope="class")
    def x(self, num):
        return np.linspace(0, 6, num=num) * u.deg

    @pytest.fixture(scope="class")
    def y(self, num, x):
        return (np.exp(-(x.value**2)) + 0.1) * u.m

    @pytest.fixture(scope="class")
    def extra_args(self):
        return {"k": 3, "ext": 0, "check_finite": False}

    @pytest.fixture(scope="class")
    def ispline_cls(self):
        return spline.LSQUnivariateSplinewithUnits

    @pytest.fixture(scope="class")
    def t(self, ispline_cls, x, y, extra_args):
        spl = spline.InterpolatedUnivariateSplinewithUnits(
            x,
            y,
            w=None,
            bbox=[None] * 2,
            **extra_args,
        )
        return spl.get_knots().value[1:-1]

    @pytest.fixture(
        scope="class",
        params=(
            {"w": None, "bbox": [None, None]},
            {"w": ..., "bbox": [None, None]},
            {"w": None, "bbox": ...},
        ),
    )
    def spls(self, request, ispline_cls, x, y, w, t, bbox, extra_args):
        _w, _bbox = request.param.values()
        w = w if _w is ... else _w
        bbox = bbox if _bbox is ... else _bbox

        return ispline_cls(x, y, t, w=w, bbox=bbox, **extra_args)

    # ===============================================================
    # Method tests

    def test_fail_init(self, ispline_cls, x, y, t):
        """Test a failed initialization b/c wrong units."""
        bad_unit = x.unit / u.m

        with pytest.raises(u.UnitConversionError):
            ispline_cls(x, y, t, bbox=[0 * bad_unit, 180 * bad_unit])
