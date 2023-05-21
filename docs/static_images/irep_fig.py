# Licensed under a 3-clause BSD style license - see LICENSE.rst

import astropy.coordinates as coord
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

import interpolated_coordinates as icoord

# Making a Representation
num = 40
x = np.linspace(0, 1, num=num) * u.kpc
rep = coord.CartesianRepresentation(
    x=x,
    y=8 / (x.value**2 + 4) * x.unit,
    z=np.linspace(2, 3, num=num) * u.kpc,
    differentials=coord.CartesianDifferential(
        d_x=np.linspace(3, 4, num=num) * (u.km / u.s),
        d_y=np.linspace(4, 5, num=num) * (u.km / u.s),
        d_z=np.linspace(5, 6, num=num) * (u.km / u.s),
    ),
)

# Now interpolating
affine = np.linspace(0, 10, num=num) * u.Myr
irep = icoord.InterpolatedRepresentation(rep, affine)

# Plotting
fig = plt.figure()
ax = fig.add_subplot()
ax.set_xlabel(r"$x$ [kpc]", fontsize=12)
ax.set_ylabel(r"$(y * z)^4$  [kpc$^8$]", fontsize=12)

plt.scatter(rep.x, (rep.y * rep.z) ** 3, label="rep points")

afn = np.linspace(0, 10, num=num * 100) * u.Myr
r = irep(afn)
plt.plot(r.x, (r.y * r.z) ** 3, zorder=-1, label="spline eval", c="k")

ax.legend()
fig.savefig("irep.png")

plt.close("all")
