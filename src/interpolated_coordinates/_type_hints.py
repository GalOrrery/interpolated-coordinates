"""Type hints.

This project extensively uses :mod:`~typing` hints.
Note that this is not (necessarily) static typing.
"""

__all__ = [
    "ArrayLike",
    # coordinates
    "CoordinateType",
    "FrameLikeType",
    # units
    "UnitType",
    "UnitLikeType",
]

__credits__ = ["Astropy"]

from typing import Union

import astropy.coordinates as coord
import astropy.units as u
import numpy as np

# -------------------------------------
# NumPy types

ArrayLike = Union[float, np.ndarray]  # np.generic isn't compatible


# -------------------------------------
# Astropy types

RepLikeType = Union[coord.BaseRepresentation, str]

CoordinateType = Union[coord.BaseCoordinateFrame, coord.SkyCoord]
"""|Frame| or |SkyCoord|"""

FrameLikeType = Union[CoordinateType, str]
"""|Frame| or |SkyCoord| or `str`"""

UnitType = Union[u.UnitBase, u.FunctionUnitBase]
"""|Unit| or :class:`~astropy.units.FunctionUnitBase`"""

UnitLikeType = Union[UnitType, str]
"""|Unit| or :class:`~astropy.units.FunctionUnitBase` or str"""
