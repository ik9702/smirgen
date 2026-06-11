"""Spherical <-> Cartesian helpers.

Ported from ``mysph2cart.m`` / ``mycart2sph.m``.

Note: this package uses the *physics* convention where the second angle is the
**inclination** measured from the +z axis (theta), not the elevation from the
xy-plane.  This matches the original SMIR generator (despite ``mycart2sph``'s
docstring calling it "elevation", it actually returns ``acos(z/r)``).
"""

import numpy as np


def sph2cart(az, inc, r):
    """Spherical (azimuth, inclination, radius) -> Cartesian (x, y, z).

    Parameters
    ----------
    az : array_like
        Azimuth in radians, counter-clockwise from +x axis (phi).
    inc : array_like
        Inclination in radians, from +z axis (theta).
    r : array_like
        Radius.
    """
    az = np.asarray(az, dtype=float)
    inc = np.asarray(inc, dtype=float)
    r = np.asarray(r, dtype=float)
    z = r * np.cos(inc)
    rcosinc = r * np.sin(inc)
    x = rcosinc * np.cos(az)
    y = rcosinc * np.sin(az)
    return x, y, z


def cart2sph(x, y, z):
    """Cartesian (x, y, z) -> Spherical (azimuth, inclination, radius).

    Returns ``(az, inc, r)`` where ``inc = acos(z / r)`` is the inclination
    from the +z axis.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    hypotxy = np.hypot(x, y)
    r = np.hypot(hypotxy, z)
    with np.errstate(invalid="ignore", divide="ignore"):
        inc = np.arccos(np.where(r == 0, 1.0, z / r))
    az = np.arctan2(y, x)
    return az, inc, r
