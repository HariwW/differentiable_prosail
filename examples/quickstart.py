"""Run one PROSAIL simulation and save the canopy reflectance spectrum."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import prosail


reflectance = prosail.run_prosail(
    n=1.5,
    cab=40.0,
    car=8.0,
    cbrown=0.0,
    cw=0.015,
    cm=0.009,
    lai=3.0,
    lidfa=50.0,
    hspot=0.1,
    tts=30.0,
    tto=0.0,
    psi=0.0,
    prospect_version="5",
    typelidf=2,
    factor="SDR",
    rsoil=1.0,
    psoil=0.5,
)

wavelength_nm = np.arange(400, 2501)
output_dir = Path(__file__).resolve().parents[1] / "output"
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "prosail_spectrum.png"

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

print(f"Generated {reflectance.size} reflectance values (400-2500 nm).")
print(f"Reflectance range: {reflectance.min():.4f} to {reflectance.max():.4f}")
print(f"Saved plot to: {output_path}")
