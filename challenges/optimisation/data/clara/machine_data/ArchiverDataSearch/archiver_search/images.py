"""Render an image out of a capture file to PNG.

Capture files sit on the server's filesystem, not the client's, so the render
happens here and the browser only ever sees a PNG. The figure is built through
the object API rather than ``pyplot``: endpoints run in a thread pool, and
pyplot's global figure registry is neither thread-safe nor self-cleaning.
"""

from __future__ import annotations

import io
from pathlib import Path

import h5py
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from . import schema

#: Long edge of the rendered figure, in inches, at 100 dpi.
FIGURE_INCHES = 6.0
DPI = 100


class ImageError(Exception):
    """The capture file is present but the requested image is not usable."""


def read_capture_image(path: Path, index: int = 1) -> np.ndarray:
    """Read the ``index``-th image (1-based) out of a capture file."""
    name = schema.image_dataset_name(index)
    try:
        with h5py.File(path, "r") as f:
            if name not in f:
                raise ImageError(f"{path.name} has no dataset {name}")
            data = np.asarray(f[name][()])
    except OSError as exc:
        raise ImageError(f"{path.name} is not readable as HDF5") from exc
    if data.ndim not in (2, 3):
        raise ImageError(f"{name} is {data.ndim}-D; expected a 2-D or RGB image")
    return data


def render_png(data: np.ndarray) -> bytes:
    """``imshow`` the array and return the figure as PNG bytes."""
    height, width = data.shape[:2]
    # Match the figure to the image's aspect ratio so imshow fills it exactly
    # and the axes can take up the whole canvas.
    if width >= height:
        figsize = (FIGURE_INCHES, FIGURE_INCHES * height / max(width, 1))
    else:
        figsize = (FIGURE_INCHES * width / max(height, 1), FIGURE_INCHES)

    figure = Figure(figsize=figsize, dpi=DPI)
    FigureCanvasAgg(figure)
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axes.imshow(data)
    axes.set_axis_off()

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    return buffer.getvalue()
