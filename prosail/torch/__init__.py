"""Differentiable and batched PyTorch implementation of PROSAIL.

The original :mod:`prosail` functions remain NumPy based. Import this module
when gradients or batched tensor execution are required::

    import prosail.torch as tprosail

The public ``run_prosail`` signature follows the original NumPy package. Each
physical parameter may be a scalar tensor, a ``[batch]`` tensor, or a
``[batch, 1]`` tensor. The result is always ``[batch, wavelengths]``.
"""

from __future__ import annotations

from typing import Iterable

import torch

from .prospect_d import run_prospect as _run_prospect_impl
from .sail_model import run_prosail as _run_prosail_impl
from .sail_model import run_sail, run_thermal_sail
from .sensor import Sentinel2Sensor


def _reference_tensor(values: Iterable[object]) -> torch.Tensor:
    for value in values:
        if isinstance(value, torch.Tensor):
            dtype = value.dtype if value.is_floating_point() else torch.get_default_dtype()
            return torch.empty((), dtype=dtype, device=value.device)
    # Match the NumPy implementation's precision when all inputs are Python scalars.
    return torch.empty((), dtype=torch.float64)


def _column(value: object, reference: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=reference.device)
        if not tensor.is_floating_point():
            tensor = tensor.to(dtype=reference.dtype)
    else:
        tensor = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)

    if tensor.ndim == 0:
        return tensor.reshape(1, 1)
    if tensor.ndim == 1:
        return tensor.reshape(-1, 1)
    if tensor.ndim == 2 and tensor.shape[1] == 1:
        return tensor
    raise ValueError("Each PROSAIL parameter must be scalar, [batch], or [batch, 1].")


def _columns(*values: object) -> tuple[torch.Tensor, ...]:
    reference = _reference_tensor(values)
    columns = [_column(value, reference) for value in values]
    return tuple(torch.broadcast_tensors(*columns))


def run_prospect(
    n,
    cab,
    car,
    cbrown,
    cw,
    cm,
    ant=0.0,
    prot=0.0,
    cbc=0.0,
    prospect_version="5",
    **kwargs,
):
    """Run differentiable PROSPECT-5, PROSPECT-D, or PROSPECT-PRO."""

    n, cab, car, cbrown, cw, cm, ant, prot, cbc = _columns(
        n, cab, car, cbrown, cw, cm, ant, prot, cbc
    )
    return _run_prospect_impl(
        N=n,
        cab=cab,
        car=car,
        cbrown=cbrown,
        cw=cw,
        cm=cm,
        ant=ant,
        prot=prot,
        cbc=cbc,
        prospect_version=prospect_version,
        **kwargs,
    )


def run_prosail(
    n,
    cab,
    car,
    cbrown,
    cw,
    cm,
    lai,
    lidfa,
    hspot,
    tts,
    tto,
    psi,
    ant=0.0,
    prot=0.0,
    cbc=0.0,
    prospect_version="5",
    typelidf=2,
    lidfb=0.0,
    factor="SDR",
    rsoil0=None,
    rsoil=1.0,
    psoil=1.0,
    **kwargs,
):
    """Run differentiable PROSAIL with an API matching ``prosail.run_prosail``.

    Gradients are preserved for every tensor parameter. Python numeric values
    are accepted as constants. ``typelidf=2`` (Campbell ellipsoidal leaf-angle
    distribution) is the differentiable, batch-capable implementation.
    """

    values = _columns(
        n,
        cab,
        car,
        cbrown,
        cw,
        cm,
        lai,
        lidfa,
        hspot,
        rsoil,
        psoil,
        tts,
        tto,
        psi,
        ant,
        prot,
        cbc,
    )
    (
        n,
        cab,
        car,
        cbrown,
        cw,
        cm,
        lai,
        lidfa,
        hspot,
        rsoil,
        psoil,
        tts,
        tto,
        psi,
        ant,
        prot,
        cbc,
    ) = values

    if rsoil0 is not None:
        rsoil0 = torch.as_tensor(rsoil0, dtype=n.dtype, device=n.device)
        if rsoil0.ndim == 1:
            rsoil0 = rsoil0.unsqueeze(0)

    return _run_prosail_impl(
        N=n,
        cab=cab,
        car=car,
        cbrown=cbrown,
        cw=cw,
        cm=cm,
        lai=lai,
        lidfa=lidfa,
        hspot=hspot,
        rsoil=rsoil,
        psoil=psoil,
        tts=tts,
        tto=tto,
        psi=psi,
        ant=ant,
        prot=prot,
        cbc=cbc,
        prospect_version=prospect_version,
        typelidf=torch.as_tensor(typelidf, device=n.device),
        lidfb=lidfb,
        factor=factor,
        rsoil0=rsoil0,
        **kwargs,
    )


__all__ = [
    "Sentinel2Sensor",
    "run_prospect",
    "run_prosail",
    "run_sail",
    "run_thermal_sail",
]
