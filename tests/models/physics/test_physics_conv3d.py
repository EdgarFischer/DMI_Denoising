import pytest
import torch

from denoising.models.physics import BaselineFreeBasisDecoder, PhysicsConv3D


def _model(spectral_axis: int = 2) -> PhysicsConv3D:
    generator = torch.Generator().manual_seed(4)
    basis = torch.complex(
        torch.randn(3, 16, generator=generator),
        torch.randn(3, 16, generator=generator),
    )
    decoder = BaselineFreeBasisDecoder(basis, 1 / 2000)
    return PhysicsConv3D(
        decoder,
        input_n_timepoints=16,
        spectral_axis=spectral_axis,
        features=(4, 8),
        spectral_strides=(2, 2),
        parameter_head_hidden_channels=12,
    )


@pytest.mark.parametrize(
    "spectral_axis,input_shape",
    [(0, (2, 2, 16, 5, 6)), (1, (2, 2, 5, 16, 6)), (2, (2, 2, 5, 6, 16))],
)
def test_reconstruction_preserves_input_layout(spectral_axis, input_shape):
    model = _model(spectral_axis)
    x = torch.randn(input_shape)
    output = model(x, return_parameters=True)

    assert output.reconstruction.shape == x.shape
    assert output.raw_parameter_maps.shape == (2, 8, 5, 6)
    assert output.parameters.amplitudes.shape == (2, 5, 6, 3)


def test_end_to_end_gradients_reach_encoder_and_parameter_head():
    model = _model(2)
    x = torch.randn(2, 2, 4, 5, 16)
    reconstruction = model(x)
    reconstruction.square().mean().backward()

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable
    assert all(parameter.grad is not None for parameter in trainable)
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainable)
    assert not model.physical_decoder.basis_fids.requires_grad


def test_model_contains_no_baseline_state():
    model = _model(2)
    assert not any("baseline" in name.lower() for name in model.state_dict())


def test_ppm_window_preserves_input_outside_selected_range():
    generator = torch.Generator().manual_seed(5)
    basis = torch.complex(
        torch.randn(2, 16, generator=generator),
        torch.randn(2, 16, generator=generator),
    )
    decoder = BaselineFreeBasisDecoder(basis, 1 / 1600)
    model = PhysicsConv3D(
        decoder,
        input_n_timepoints=16,
        spectral_axis=2,
        features=(4, 8),
        spectral_strides=(2, 2),
        parameter_head_hidden_channels=12,
        denoising_ppm_range=(4.4, 4.9),
        ppm_reference=4.65,
        hz_per_ppm=100.0,
    )
    x = torch.randn(1, 2, 3, 4, 16)
    output = model(x)
    outside = ~model.denoising_frequency_mask
    torch.testing.assert_close(output[..., outside], x[..., outside])
    assert torch.any(model.denoising_frequency_mask)
    assert torch.any(outside)
