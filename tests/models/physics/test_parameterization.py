import torch

from denoising.models.physics import MinimalPhysicalParameterization
from denoising.models.physics.parameterization import (
    LCModelKernelParameterization,
    StandardizedLCModelKernelParameterization,
    StandardizedPhysicalParameterization,
)


def test_only_physically_nonnegative_parameters_are_transformed():
    layer = MinimalPhysicalParameterization(n_basis_components=2)
    raw = torch.tensor(
        [[[-2.0], [3.0], [-4.0], [-5.0], [6.0], [-7.0], [8.0]]]
    ).reshape(1, 7, 1, 1)
    parameters = layer(raw)

    assert torch.all(parameters.amplitudes > 0)
    assert torch.all(parameters.lorentzian_fwhm_hz > 0)
    assert torch.all(parameters.gaussian_fwhm_hz > 0)
    torch.testing.assert_close(parameters.frequency_shift_hz, raw[:, 2])
    torch.testing.assert_close(parameters.zero_order_phase_radians, raw[:, 5])
    torch.testing.assert_close(parameters.first_order_phase_rad_per_hz, raw[:, 6])
    torch.testing.assert_close(
        parameters.lorentzian_fwhm_hz,
        torch.nn.functional.softplus(raw[:, 3]),
    )
    torch.testing.assert_close(
        parameters.gaussian_fwhm_hz,
        torch.nn.functional.softplus(raw[:, 4]),
    )
    torch.testing.assert_close(
        parameters.amplitudes,
        torch.nn.functional.softplus(raw[:, :2]).movedim(1, -1),
    )


def test_parameterization_has_no_parameters_or_buffers():
    layer = MinimalPhysicalParameterization(n_basis_components=3)
    assert list(layer.parameters()) == []
    assert layer.state_dict() == {}


def test_fwhm_transform_is_positive_unbounded_with_nonvanishing_large_gradient():
    raw_values = torch.tensor([2.0, 5.0, 10.0], requires_grad=True)
    transformed = torch.nn.functional.softplus(raw_values)
    gradients = torch.autograd.grad(transformed.sum(), raw_values)[0]

    assert transformed[0] < transformed[1] < transformed[2]
    assert torch.all(gradients > 0)
    assert gradients[-1] > 0.99


def test_standardized_parameterization_decodes_mean_and_preserves_constraints():
    means = (0.2, 0.1, 4.0, 12.0, 8.0, -0.2, 0.001)
    stds = (0.05, 0.02, 2.0, 3.0, 2.0, 0.4, 0.0002)
    layer = StandardizedPhysicalParameterization(2, means, stds)
    raw = layer.raw_at_population_mean()[None, :, None, None]
    physical = layer.physical_channels(raw)
    z = layer.standardized_maps(raw)

    torch.testing.assert_close(
        physical[:, :, 0, 0], torch.tensor(means)[None], rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(z, torch.zeros_like(z), rtol=1e-5, atol=1e-5)
    assert torch.all(physical[:, :2] > 0)
    assert torch.all(physical[:, 3:5] > 0)


def test_standardized_physical_roundtrip_has_finite_gradients():
    layer = StandardizedPhysicalParameterization(
        1, (0.2, 3.0, 12.0, 8.0, -0.2, 0.001),
        (0.05, 2.0, 3.0, 2.0, 0.4, 0.0002),
    )
    raw = torch.randn(2, 6, 3, 4, requires_grad=True)
    physical = layer.physical_channels(raw)
    z_from_physical = layer.standardize_physical_channels(physical)
    torch.testing.assert_close(z_from_physical, layer.standardized_maps(raw))
    z_from_physical.square().mean().backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()


def test_teacher_conversion_scales_only_amplitudes():
    layer = StandardizedPhysicalParameterization(
        2,
        (0.6, 0.3, 4.0, 12.0, 8.0, -0.2, 0.001),
        (0.15, 0.06, 2.0, 3.0, 2.0, 0.4, 0.0002),
        teacher_to_model_amplitude_scale=3.0,
    )
    teacher = torch.tensor(
        [[0.2, 0.1, 4.0, 12.0, 8.0, -0.2, 0.001]]
    )[:, :, None, None]
    converted = layer.convert_teacher_physical_channels(teacher)
    torch.testing.assert_close(
        converted[:, :, 0, 0],
        torch.tensor([[0.6, 0.3, 4.0, 12.0, 8.0, -0.2, 0.001]]),
    )


def test_lcmodel_kernel_parameterization_constraints_and_shapes():
    layer = LCModelKernelParameterization(
        3, lineshape_kernel_size=23,
    )
    raw = torch.randn(2, layer.n_output_parameters, 4, 5)
    parameters = layer(raw)
    assert parameters.amplitudes.shape == (2, 4, 5, 3)
    assert parameters.metabolite_frequency_shift_hz.shape == (2, 4, 5, 3)
    assert parameters.metabolite_lorentzian_fwhm_hz.shape == (2, 4, 5, 3)
    assert parameters.lineshape_kernel.shape == (2, 4, 5, 23)
    assert torch.all(parameters.amplitudes > 0)
    assert torch.all(parameters.metabolite_lorentzian_fwhm_hz > 0)
    torch.testing.assert_close(
        parameters.metabolite_frequency_shift_hz,
        raw[:, layer._sections["metabolite_shifts"]].movedim(1, -1),
    )
    torch.testing.assert_close(
        parameters.lineshape_kernel.sum(-1),
        torch.ones_like(parameters.lineshape_kernel[..., 0]),
    )


def test_lcmodel_lineshape_coordinates_are_standardized_at_initial_point():
    layer = LCModelKernelParameterization(
        2,
        metabolite_shift_mean_hz=0.0,
        metabolite_shift_std_hz=1.0,
        metabolite_fwhm_mean_hz=5.0,
        metabolite_fwhm_std_hz=2.5,
    )
    raw = layer.raw_at_initial_values(torch.tensor([0.2, 0.3]), 5.0)
    raw = raw[None, :, None, None].requires_grad_()
    parameters = layer(raw)

    torch.testing.assert_close(
        parameters.metabolite_frequency_shift_hz,
        torch.zeros(1, 1, 1, 2), atol=1e-6, rtol=0,
    )
    torch.testing.assert_close(
        parameters.metabolite_lorentzian_fwhm_hz,
        torch.full((1, 1, 1, 2), 5.0), atol=1e-5, rtol=0,
    )
    shift_grad = torch.autograd.grad(
        parameters.metabolite_frequency_shift_hz.sum(), raw, retain_graph=True
    )[0]
    fwhm_grad = torch.autograd.grad(
        parameters.metabolite_lorentzian_fwhm_hz.sum(), raw
    )[0]
    torch.testing.assert_close(
        shift_grad[:, layer._sections["metabolite_shifts"]],
        torch.ones(1, 2, 1, 1), atol=1e-6, rtol=0,
    )
    torch.testing.assert_close(
        fwhm_grad[:, layer._sections["metabolite_lorentz"]],
        torch.full((1, 2, 1, 1), 2.5), atol=1e-5, rtol=0,
    )


def test_standardized_lcmodel_kernel_uses_old_scales_only_for_retained_parameters():
    n = 2
    means = (0.2, 0.4, 3.0, 12.0, 8.0, -0.1, 0.002)
    stds = (0.05, 0.1, 2.0, 4.0, 3.0, 0.5, 0.001)
    layer = StandardizedLCModelKernelParameterization(
        n, means, stds, lineshape_kernel_size=23
    )
    raw = layer.raw_at_initial_values(torch.ones(n), 5.0)[None, :, None, None]
    parameters = layer(raw)
    torch.testing.assert_close(parameters.amplitudes[0, 0, 0], torch.tensor(means[:n]))
    torch.testing.assert_close(parameters.frequency_shift_hz, torch.full((1, 1, 1), means[n]))
    torch.testing.assert_close(parameters.zero_order_phase_radians, torch.full((1, 1, 1), means[n + 3]))
    torch.testing.assert_close(parameters.first_order_phase_rad_per_hz, torch.full((1, 1, 1), means[n + 4]))
    torch.testing.assert_close(
        parameters.metabolite_lorentzian_fwhm_hz,
        torch.full((1, 1, 1, n), 5.0),
    )
    torch.testing.assert_close(
        parameters.metabolite_frequency_shift_hz,
        torch.zeros(1, 1, 1, n),
    )
    torch.testing.assert_close(
        parameters.lineshape_kernel.sum(-1), torch.ones(1, 1, 1)
    )
