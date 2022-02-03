# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

# THIRD PARTY
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import quantity_support

# LOCAL
from interpolated_coordinates.utils import InterpolatedUnivariateSplinewithUnits

quantity_support()

x = np.linspace(-3, 3, num=50) * u.s
y = 8 * u.m / (x.value ** 2 + 4)
spl = InterpolatedUnivariateSplinewithUnits(x, y)
spl(np.linspace(-2, 2, num=10) * u.s)  # Evaluate spline

xs = np.linspace(-3, 3, num=1000) * u.s  # for sampling

fig = plt.figure(figsize=(7, 4))
ax = fig.add_subplot(111)
ax.plot(xs, spl(xs), c="gray", alpha=0.7, lw=3, label="evaluated spline")
ax.scatter(x, y, c="r", s=25, label="points")

ax.set_title("Witch of Agnesi (a=1)")
ax.set_xlabel(f"x [{ax.get_xlabel()}]")
ax.set_ylabel(f"y [{ax.get_ylabel()}]")
plt.legend()
fig.savefig("spline.png")

plt.close("all")
