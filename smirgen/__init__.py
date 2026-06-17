"""smirgen - Spherical Microphone array Impulse Response generator (Python).

Python port of the SMIR generator from the International Audio Laboratories
Erlangen. The performance-critical image-source loop runs in a native
pybind11 extension (``_smir_loop``); everything else is NumPy/SciPy.

Example
-------
>>> import numpy as np
>>> from smirgen import smir_generator
>>> h, H, beta_hat = smir_generator(
...     c=343, procFs=8000, sphLocation=[1.6, 4.05, 1.7], s=[3.37, 4.05, 1.7],
...     L=[5, 6, 4], beta=0.3, sphType="rigid", sphRadius=0.042,
...     mic=[[np.pi/4, np.pi], [np.pi/2, np.pi]], N_harm=30,
...     nsample=2048, K=1, order=6, HP=1)
"""

from .generator import smir_generator, smir_generator_batch, order_per_freq
from .coords import sph2cart, cart2sph
from .plotting import plot_geometry
from .analysis import relative_db, direct_reference
from .torch_gen import SmirArray            # GPU backend (torch imported lazily)

__all__ = ["smir_generator", "smir_generator_batch", "order_per_freq",
           "sph2cart", "cart2sph", "plot_geometry", "relative_db",
           "direct_reference", "SmirArray"]
__version__ = "2.8.0"
