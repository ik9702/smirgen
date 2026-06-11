"""3D geometry visualisation for the SMIR generator setup.

Draws the shoebox room, the spherical microphone array (centre, sphere surface
and individual microphones), the source, and the source look direction.

The standalone :func:`plot_geometry` can be used on its own to preview a setup
before running a (potentially slow) simulation. ``smir_generator(...,
plot_geometry=True)`` is a thin convenience wrapper around it.
"""

import numpy as np

from .coords import sph2cart

__all__ = ["plot_geometry"]


def _room_edges(L):
    """Return the 12 edges of the shoebox room [0,Lx]x[0,Ly]x[0,Lz]."""
    Lx, Ly, Lz = L
    v = np.array([[0, 0, 0], [Lx, 0, 0], [Lx, Ly, 0], [0, Ly, 0],
                  [0, 0, Lz], [Lx, 0, Lz], [Lx, Ly, Lz], [0, Ly, Lz]], float)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),      # floor
             (4, 5), (5, 6), (6, 7), (7, 4),      # ceiling
             (0, 4), (1, 5), (2, 6), (3, 7)]      # pillars
    return [(v[a], v[b]) for a, b in edges]


def _mic_positions(sphLocation, sphRadius, mic):
    """Absolute microphone coordinates from (azimuth, inclination) angles."""
    mic = np.asarray(mic, float)
    if mic.ndim == 1:
        mic = mic.reshape(1, -1)
    mx, my, mz = sph2cart(mic[:, 0], mic[:, 1], 1.0)
    return sphLocation + sphRadius * np.column_stack([mx, my, mz])


def _draw_array(ax, sphLocation, sphRadius, mic, label_mics=False):
    """Draw the array centre, sphere surface and microphones onto ``ax``."""
    ax.scatter(*sphLocation, color="tab:blue", s=60, marker="o",
               label="Array centre")
    if sphRadius > 0:
        u = np.linspace(0, 2 * np.pi, 24)
        v = np.linspace(0, np.pi, 16)
        xs = sphLocation[0] + sphRadius * np.outer(np.cos(u), np.sin(v))
        ys = sphLocation[1] + sphRadius * np.outer(np.sin(u), np.sin(v))
        zs = sphLocation[2] + sphRadius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(xs, ys, zs, color="tab:blue", alpha=0.15, lw=0.5)
    if mic is not None:
        mpos = _mic_positions(sphLocation, sphRadius, mic)
        ax.scatter(mpos[:, 0], mpos[:, 1], mpos[:, 2], color="tab:green",
                   s=40, marker="^", depthshade=False,
                   label=f"Microphones (M={mpos.shape[0]})")
        if label_mics:
            for i, p in enumerate(mpos):
                ax.text(p[0], p[1], p[2], f" {i}", fontsize=8,
                        color="tab:green")
        return mpos
    return None


def plot_geometry(L, sphLocation, s, sphRadius=0.0, mic=None, src_ang=None,
                  sphType="rigid", ax=None, show=False, title=None,
                  array_inset=True):
    """Plot the room / array / source geometry in 3D.

    Parameters
    ----------
    L : (3,) array_like
        Room dimensions (x, y, z) in m.
    sphLocation : (3,) array_like
        Array centre (x, y, z) in m.
    s : (3,) array_like
        Source position (x, y, z) in m.
    sphRadius : float, optional
        Array radius (m). ``0`` draws a single point at the centre.
    mic : (M, 2) array_like, optional
        Microphone angles ``(azimuth, inclination)`` in radians. The markers are
        placed on the sphere of radius ``sphRadius`` around ``sphLocation``.
    src_ang : (2,) array_like, optional
        Source look direction ``(azimuth, inclination)`` in radians, drawn as an
        arrow from the source.
    sphType : {'open', 'rigid'}, optional
        Only affects the legend / sphere styling.
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D, optional
        Existing 3D axes to draw into. A new figure is created if omitted.
        Passing ``ax`` disables the array close-up panel.
    show : bool, optional
        Call ``plt.show()`` before returning.
    title : str, optional
        Title for the room panel.
    array_inset : bool, optional
        Add a second panel zoomed onto the array so the (typically centimetre
        scale) microphone layout is visible inside the metre-scale room. Only
        used when ``ax`` is None, ``sphRadius > 0`` and ``mic`` is given.

    Returns
    -------
    fig, ax : matplotlib Figure and the room-panel 3D Axes. The array close-up
        panel, when present, is available as ``fig.axes[1]``.
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d)
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "plot_geometry requires matplotlib. Install it with "
            "`pip install matplotlib`.") from exc

    L = np.asarray(L, float).reshape(3)
    sphLocation = np.asarray(sphLocation, float).reshape(3)
    s = np.asarray(s, float).reshape(3)
    if src_ang is not None:
        src_ang = np.asarray(src_ang, float).reshape(-1)

    want_inset = (ax is None and array_inset and sphRadius > 0
                  and mic is not None)

    if ax is None:
        if want_inset:
            fig = plt.figure(figsize=(13, 6))
            ax = fig.add_subplot(121, projection="3d")
            ax_zoom = fig.add_subplot(122, projection="3d")
        else:
            fig = plt.figure(figsize=(8, 7))
            ax = fig.add_subplot(111, projection="3d")
            ax_zoom = None
    else:
        fig = ax.figure
        ax_zoom = None

    # --- room panel --------------------------------------------------------
    for p0, p1 in _room_edges(L):
        ax.plot(*zip(p0, p1), color="0.6", lw=1.0)

    _draw_array(ax, sphLocation, sphRadius, mic)

    ax.scatter(*s, color="tab:red", s=80, marker="*", label="Source")
    ax.plot(*zip(s, sphLocation), color="tab:red", ls="--", lw=1.0, alpha=0.6)

    if src_ang is not None:
        lx, ly, lz = sph2cart(src_ang[0], src_ang[1], 1.0)
        arrow_len = 0.25 * float(np.linalg.norm(L))
        ax.quiver(s[0], s[1], s[2], lx, ly, lz, length=arrow_len,
                  color="tab:orange", lw=2.0, label="Source look dir.")

    ax.set_xlim(0, L[0])
    ax.set_ylim(0, L[1])
    ax.set_zlim(0, L[2])
    try:
        ax.set_box_aspect(L)              # true-to-scale (matplotlib >= 3.3)
    except Exception:                     # pragma: no cover
        pass
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(title or f"Room ({sphType} sphere, r={sphRadius} m)")
    ax.legend(loc="upper left", fontsize=8)

    # --- array close-up panel ---------------------------------------------
    if want_inset:
        _draw_array(ax_zoom, sphLocation, sphRadius, mic, label_mics=True)
        pad = 1.6 * sphRadius
        ax_zoom.set_xlim(sphLocation[0] - pad, sphLocation[0] + pad)
        ax_zoom.set_ylim(sphLocation[1] - pad, sphLocation[1] + pad)
        ax_zoom.set_zlim(sphLocation[2] - pad, sphLocation[2] + pad)
        try:
            ax_zoom.set_box_aspect((1, 1, 1))
        except Exception:                 # pragma: no cover
            pass
        ax_zoom.set_xlabel("x (m)")
        ax_zoom.set_ylabel("y (m)")
        ax_zoom.set_zlabel("z (m)")
        ax_zoom.set_title("Array close-up (mics numbered)")
        ax_zoom.legend(loc="upper left", fontsize=8)

    if show:
        plt.show()
    return fig, ax
