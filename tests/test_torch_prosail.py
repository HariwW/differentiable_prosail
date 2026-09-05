import numpy as np
import torch

import prosail
import prosail.torch as torch_prosail


PARAMETERS = {
    "n": 1.5,
    "cab": 40.0,
    "car": 8.0,
    "cbrown": 0.0,
    "cw": 0.01,
    "cm": 0.009,
    "lai": 3.0,
    "lidfa": 50.0,
    "hspot": 0.01,
    "tts": 30.0,
    "tto": 10.0,
    "psi": 0.0,
    "prospect_version": "5",
    "typelidf": 2,
    "rsoil": 1.0,
    "psoil": 0.5,
    "factor": "SDR",
}


def check_close(name, actual, expected, rtol=1e-5, atol=1e-6):
    """比较两个数组，并打印误差。"""
    actual = np.asarray(actual)
    expected = np.asarray(expected)

    abs_error = np.abs(actual - expected)

    print(f"\n{name}")
    print(f"  shape       : {actual.shape}")
    print(f"  max abs err : {abs_error.max():.6e}")
    print(f"  mean abs err: {abs_error.mean():.6e}")

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
    )

    print("  ✓ PASS")


def test_prosail_numpy_vs_torch():
    print("\n" + "=" * 70)
    print("1. 测试 Torch PROSAIL 是否与 NumPy PROSAIL 一致")
    print("=" * 70)

    expected = prosail.run_prosail(**PARAMETERS)

    actual = torch_prosail.run_prosail(**PARAMETERS)

    print("NumPy shape :", expected.shape)
    print("Torch shape :", actual.shape)

    assert actual.shape == (1, 2101)

    actual_np = actual.detach().cpu().numpy()[0]

    check_close(
        "PROSAIL spectrum",
        actual_np,
        expected,
    )

    print("\n前 10 个波长反射率：")
    for wavelength, ref, pred in zip(
        range(400, 410),
        expected[:10],
        actual_np[:10],
    ):
        print(
            f"  {wavelength} nm: "
            f"numpy={ref:.8f}, torch={pred:.8f}"
        )


def test_batch_and_gradient():
    print("\n" + "=" * 70)
    print("2. 测试 batch 输入 + 自动微分")
    print("=" * 70)

    lai = torch.tensor(
        [1.0, 3.0, 5.0],
        dtype=torch.float64,
        requires_grad=True,
    )

    cab = torch.tensor(
        [30.0, 40.0, 60.0],
        dtype=torch.float64,
        requires_grad=True,
    )

    spectra = torch_prosail.run_prosail(
        **{
            **PARAMETERS,
            "lai": lai,
            "cab": cab,
        }
    )

    print("PROSAIL spectra shape:", spectra.shape)

    sensor = torch_prosail.Sentinel2Sensor()

    bands = sensor(spectra)

    print("Sentinel-2 bands shape:", bands.shape)

    print("\nSentinel-2 reflectance:")
    print(bands.detach().cpu().numpy())

    # 随便构造一个标量 loss
    loss = bands.sum()

    print("\nloss =", loss.item())

    # 反向传播
    loss.backward()

    print("\nLAI gradient:")
    print(lai.grad)

    print("\nCab gradient:")
    print(cab.grad)

    assert spectra.shape == (3, 2101)
    assert bands.shape == (3, 10)

    assert torch.isfinite(spectra).all()
    assert torch.isfinite(bands).all()

    assert torch.isfinite(lai.grad).all()
    assert torch.isfinite(cab.grad).all()

    assert not torch.allclose(
        lai.grad,
        torch.zeros_like(lai.grad),
    )

    assert not torch.allclose(
        cab.grad,
        torch.zeros_like(cab.grad),
    )

    print("\n✓ Batch + gradient PASS")


def test_autograd_vs_numerical_gradient():
    print("\n" + "=" * 70)
    print("3. 测试 Autograd 梯度 vs 数值中心差分")
    print("=" * 70)

    tests = [
        ("cab", 1e-3),
        ("cw", 1e-6),
        ("lai", 1e-4),
        ("lidfa", 1e-3),
    ]

    # 选几个波长索引
    indices = torch.tensor(
        [265, 442, 810, 1210, 1790]
    )

    for name, step in tests:

        print(f"\n参数: {name}")
        print("-" * 50)

        value = torch.tensor(
            PARAMETERS[name],
            dtype=torch.float64,
            requires_grad=True,
        )

        def objective(parameter):

            output = torch_prosail.run_prosail(
                **{
                    **PARAMETERS,
                    name: parameter,
                }
            )

            return output[0, indices].sum()

        # PyTorch 自动微分
        automatic = torch.autograd.grad(
            objective(value),
            value,
        )[0]

        # 数值中心差分
        with torch.no_grad():

            plus = objective(value + step)
            minus = objective(value - step)

            numerical = (
                plus - minus
            ) / (2 * step)

        print(
            f"Autograd gradient : "
            f"{automatic.item():.10e}"
        )

        print(
            f"Numerical gradient: "
            f"{numerical.item():.10e}"
        )

        difference = abs(
            automatic.item()
            - numerical.item()
        )

        print(
            f"Absolute error    : "
            f"{difference:.10e}"
        )

        torch.testing.assert_close(
            automatic,
            numerical,
            rtol=2e-4,
            atol=1e-7,
        )

        print("✓ PASS")


def test_zero_lai():
    print("\n" + "=" * 70)
    print("4. 测试 LAI=0 时是否返回土壤光谱")
    print("=" * 70)

    lai = torch.tensor(
        [0.0, 3.0, 0.0]
    )

    spectra = torch_prosail.run_prosail(
        **{
            **PARAMETERS,
            "lai": lai,
        }
    )

    print("spectra shape:", spectra.shape)

    print(
        "LAI=0 样本 1 和样本 3 是否相同:",
        torch.allclose(
            spectra[0],
            spectra[2],
        ),
    )

    print(
        "LAI=0 和 LAI=3 是否不同:",
        not torch.allclose(
            spectra[0],
            spectra[1],
        ),
    )

    assert spectra.shape == (3, 2101)

    torch.testing.assert_close(
        spectra[0],
        spectra[2],
    )

    assert not torch.allclose(
        spectra[0],
        spectra[1],
    )

    print("✓ Zero LAI PASS")


def test_prospect_d():
    print("\n" + "=" * 70)
    print("5. 测试 PROSPECT-D Torch vs NumPy")
    print("=" * 70)

    expected = prosail.run_prospect(
        1.5,
        40.0,
        8.0,
        0.0,
        0.01,
        0.009,
        ant=2.0,
        prospect_version="D",
    )

    actual = torch_prosail.run_prospect(
        1.5,
        40.0,
        8.0,
        0.0,
        0.01,
        0.009,
        ant=2.0,
        prospect_version="D",
    )

    names = [
        "wavelength",
        "leaf reflectance",
        "leaf transmittance",
    ]

    for i, (expected_array, actual_tensor) in enumerate(
        zip(expected, actual)
    ):

        actual_array = (
            actual_tensor
            .detach()
            .cpu()
            .numpy()
            .squeeze()
        )

        check_close(
            names[i],
            actual_array,
            expected_array,
        )


def test_sentinel2_sensor():
    print("\n" + "=" * 70)
    print("6. 测试 Sentinel-2 Sensor")
    print("=" * 70)

    sensor = torch_prosail.Sentinel2Sensor()

    print("Bands:")
    print(sensor.bands)

    print("\nweights shape:")
    print(sensor.weights.shape)

    sums = sensor.weights.sum(dim=1)

    print("\n每个波段权重和:")
    print(sums)

    torch.testing.assert_close(
        sums,
        torch.ones(
            len(sensor.bands),
            dtype=torch.float64,
        ),
    )

    print("\n✓ Sentinel-2 weights normalization PASS")

    # 再实际模拟一个光谱
    spectrum = torch_prosail.run_prosail(
        **PARAMETERS
    )

    bands = sensor(spectrum)

    print("\n模拟 Sentinel-2 reflectance:")

    for name, value in zip(
        sensor.bands,
        bands[0],
    ):
        print(
            f"  {name:>4s}: "
            f"{value.item():.6f}"
        )


def test_end_to_end_gradient():
    print("\n" + "=" * 70)
    print("7. 测试完整梯度链：参数 → PROSAIL → Sentinel2 → Loss")
    print("=" * 70)

    lai = torch.tensor(
        3.0,
        dtype=torch.float64,
        requires_grad=True,
    )

    cab = torch.tensor(
        40.0,
        dtype=torch.float64,
        requires_grad=True,
    )

    cw = torch.tensor(
        0.01,
        dtype=torch.float64,
        requires_grad=True,
    )

    spectra = torch_prosail.run_prosail(
        **{
            **PARAMETERS,
            "lai": lai,
            "cab": cab,
            "cw": cw,
        }
    )

    sensor = torch_prosail.Sentinel2Sensor()

    bands = sensor(spectra)

    # 假设目标 Sentinel-2 光谱
    target = bands.detach().clone()

    # 人为稍微扰动
    target = target + 0.01

    loss = torch.mean(
        (bands - target) ** 2
    )

    loss.backward()

    print("Loss:")
    print(loss.item())

    print("\n梯度:")
    print(
        f"dL/dLAI = "
        f"{lai.grad.item():.10e}"
    )

    print(
        f"dL/dCab = "
        f"{cab.grad.item():.10e}"
    )

    print(
        f"dL/dCw  = "
        f"{cw.grad.item():.10e}"
    )

    assert torch.isfinite(lai.grad)
    assert torch.isfinite(cab.grad)
    assert torch.isfinite(cw.grad)

    print("\n✓ End-to-end differentiability PASS")


def main():

    print("\n")
    print("#" * 70)
    print("      PROSAIL.TORCH FULL FUNCTION TEST")
    print("#" * 70)

    tests = [
        test_prosail_numpy_vs_torch,
        test_batch_and_gradient,
        test_autograd_vs_numerical_gradient,
        test_zero_lai,
        test_prospect_d,
        test_sentinel2_sensor,
        test_end_to_end_gradient,
    ]

    passed = 0
    failed = 0

    for test in tests:

        try:

            test()

            passed += 1

        except Exception as exc:

            failed += 1

            print("\n❌ TEST FAILED")

            print(
                f"{test.__name__}: "
                f"{type(exc).__name__}: {exc}"
            )

    print("\n")
    print("#" * 70)
    print("TEST SUMMARY")
    print("#" * 70)

    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n✅ ALL TESTS PASSED")
    else:
        print(
            f"\n❌ {failed} TEST(S) FAILED"
        )


if __name__ == "__main__":
    main()
