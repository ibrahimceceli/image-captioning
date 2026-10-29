"""Overlay caption attention on the exact RGB crop seen by the model."""

from collections.abc import Sequence
from pathlib import Path
import math

import numpy as np
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import torch
from torch.nn import functional as F


def visualize_attention(
    image: Image.Image | np.ndarray,
    words: Sequence[str],
    attention: torch.Tensor | np.ndarray,
    *,
    output_path: str | Path | None = None,
    columns: int = 4,
    alpha: float = 0.5,
) -> Figure:
    """Return a figure with one word/attention overlay per subplot.

    image: The *exact* preprocessed 224x224 RGB crop, before ImageNet
        normalization: a PIL RGB image or HWC NumPy array (uint8 [0,255]
        or floating point [0,1]). Tensor images are deliberately rejected.
        The caller must undo normalization explicitly if needed; this helper
        cannot establish whether a supplied crop matches the model input.
    words: Use caption.split() for text returned by generate_caption.
    attention: [len(words), 49], including an empty [0,49] for no words.
        Region index row*7+column maps to that location, with row 0 at top.
        Bilinear interpolation enlarges only the heatmap, never the image.
    output_path: Optional PNG path. Parent directory must already exist.

    All overlays share the same color range, from zero to the largest input
    weight. Empty captions produce an explanatory panel. No window is opened.
    """
    if isinstance(image, Image.Image):
        if image.mode != "RGB":
            raise ValueError("image must be RGB, not grayscale or RGBA")
        pixels = np.asarray(image)
    elif isinstance(image, np.ndarray):
        pixels = image
    else:
        raise ValueError("image must be a PIL RGB image or HWC NumPy array, not a normalized tensor")
    if pixels.shape != (224, 224, 3):
        raise ValueError("image must have shape [224, 224, 3]; no resizing or cropping is performed")
    if pixels.dtype != np.uint8:
        if not np.issubdtype(pixels.dtype, np.floating):
            raise ValueError("RGB image must use uint8 [0,255] or floating point [0,1]")
        if not np.isfinite(pixels).all() or (pixels < 0).any() or (pixels > 1).any():
            raise ValueError("Floating RGB image must be finite and in [0,1]; undo normalization first")
    if isinstance(words, (str, bytes)) or not isinstance(words, Sequence):
        raise ValueError("words must be a sequence of words; use caption.split() for text")
    if any(not isinstance(word, str) or not word or any(c.isspace() for c in word) for word in words):
        raise ValueError("Each entry in words must be one nonempty word")
    try:
        weights = torch.as_tensor(attention).detach().to(device="cpu", dtype=torch.float32)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("attention must be a numeric tensor or array shaped [word_count,49]") from error
    if tuple(weights.shape) != (len(words), 49):
        raise ValueError("attention must have shape [word_count,49] matching the words")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("attention must contain finite, nonnegative weights")
    if isinstance(columns, bool) or not isinstance(columns, int) or columns < 1:
        raise ValueError("columns must be a positive integer")
    if not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")

    count = len(words)
    ncols = min(columns, max(count, 1))
    nrows = math.ceil(max(count, 1) / ncols)
    figure = Figure(figsize=(3.5 * ncols, 3.8 * nrows), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(nrows, ncols, squeeze=False).ravel()
    for axis in axes:
        axis.set_axis_off()
    if not count:
        axes[0].text(0.5, 0.5, "Empty caption: no attention maps", ha="center", va="center", wrap=True)
    else:
        maps = F.interpolate(
            weights.reshape(count, 1, 7, 7), size=(224, 224),
            mode="bilinear", align_corners=False,
        )[:, 0].numpy()
        vmax = max(float(weights.max()), 1e-12)
        for word, heatmap, axis in zip(words, maps, axes):
            axis.imshow(pixels, origin="upper")
            axis.imshow(heatmap, origin="upper", cmap="inferno", alpha=alpha,
                        vmin=0, vmax=vmax, interpolation="nearest")
            axis.set_title(word, wrap=True)
    if output_path is not None:
        figure.savefig(output_path, format="png", dpi=150)
    return figure
