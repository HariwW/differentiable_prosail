"""Differentiable Sentinel-2 spectral response integration."""

from __future__ import annotations

from importlib.resources import files

import numpy as np
import torch
from torch import nn


SENTINEL2_BAND_INDEX = {
    "B1": 0,
    "B2": 1,
    "B3": 2,
    "B4": 3,
    "B5": 4,
    "B6": 5,
    "B7": 6,
    "B8": 7,
    "B8A": 8,
    "B9": 9,
    "B10": 10,
    "B11": 11,
    "B12": 12,
}

DEFAULT_SENTINEL2_BANDS = (
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
)


class Sentinel2Sensor(nn.Module):
    """Convert 400-2500 nm PROSAIL spectra into Sentinel-2 reflectances.

    The integration follows Appendix B of Zerah et al. (2024): PROSAIL
    reflectance is weighted by solar irradiance and the band-specific relative
    spectral response. Input shape is ``[..., 2101]`` and output shape is
    ``[..., number_of_bands]``.
    """

    def __init__(self, bands=DEFAULT_SENTINEL2_BANDS):
        super().__init__()
        self.bands = tuple(bands)
        unknown = set(self.bands) - set(SENTINEL2_BAND_INDEX)
        if unknown:
            raise ValueError(f"Unknown Sentinel-2 bands: {sorted(unknown)}")

        # This is the exact RSR/irradiance table distributed with PROSAIL-VAE,
        # retained here so the paper's sensor simulation can be reproduced.
        response_path = files("prosail.torch").joinpath("data/sentinel2.rsr")
        response = np.loadtxt(response_path, unpack=True)
        weights = np.zeros((13, 2101), dtype=np.float64)
        start = int(round(response[0, 0] * 1000))
        stop = int(round(response[0, -1] * 1000))
        left = max(400, start)
        right = min(2500, stop)
        response_slice = slice(left - start, right - start + 1)
        spectrum_slice = slice(left - 400, right - 400 + 1)
        weights[:, spectrum_slice] = (
            response[2:, response_slice] * response[1, response_slice]
        )
        indices = [SENTINEL2_BAND_INDEX[name] for name in self.bands]
        weights = weights[indices]
        weights /= weights.sum(axis=1, keepdims=True)
        self.register_buffer("weights", torch.from_numpy(weights))

    def forward(self, reflectance: torch.Tensor) -> torch.Tensor:
        if reflectance.shape[-1] != 2101:
            raise ValueError(
                "Sentinel2Sensor expects full 1 nm PROSAIL spectra with "
                "2101 samples from 400 through 2500 nm."
            )
        weights = self.weights.to(
            dtype=reflectance.dtype, device=reflectance.device
        )
        return reflectance @ weights.transpose(0, 1)
