"""Demonstrate batched PROSAIL, Sentinel-2 integration, and backpropagation."""

import torch

import prosail.torch as prosail
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# One learnable LAI value. All other quantities are fixed physical constants.
lai = torch.tensor(2.0, requires_grad=True)

canopy_spectrum = prosail.run_prosail(
    n=1.5,
    cab=40.0,
    car=8.0,
    cbrown=0.0,
    cw=0.015,
    cm=0.009,
    lai=lai,
    lidfa=50.0,
    hspot=0.1,
    tts=30.0,
    tto=0.0,
    psi=0.0,
    prospect_version="5",
    rsoil=1.0,
    psoil=0.5,
)


reflectance = canopy_spectrum.detach().squeeze(0).cpu().numpy()

wavelength_nm = np.arange(400, 2501)
output_dir = Path(__file__).resolve().parents[1] / "output"
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "prosail_spectrum_diff.png"

fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(wavelength_nm, reflectance, linewidth=1.8)
ax.set(
    title="PROSAIL canopy reflectance (LAI=3, Cab=40)",
    xlabel="Wavelength (nm)",
    ylabel="Reflectance",
    xlim=(400, 2500),
    ylim=(0, max(0.6, float(reflectance.max()) * 1.05)),
)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(output_path, dpi=160)


sensor = prosail.Sentinel2Sensor()
sentinel2_reflectance = sensor(canopy_spectrum)

# Backpropagate from the simulated Sentinel-2 near-infrared B8 reflectance.
b8_index = sensor.bands.index("B8")
b8_reflectance = sentinel2_reflectance[0, b8_index]
b8_reflectance.backward()

print(f"Full spectrum shape: {tuple(canopy_spectrum.shape)}")
print(f"Sentinel-2 shape: {tuple(sentinel2_reflectance.shape)}")
print(f"B8 reflectance: {b8_reflectance.detach().item():.6f}")
print(f"d(B8)/d(LAI): {lai.grad.item():.6f}")
print("Sentinel-2 bands:")
for name, value in zip(sensor.bands, sentinel2_reflectance[0].detach()):
    print(f"  {name:>3}: {value.item():.6f}")
