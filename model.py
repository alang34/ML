import math
import torch
from torch import nn
from torch.nn import functional as F

class PINN(nn.Module):
    velocity_representation = "scaled_vector_potential_v1"

    def __init__(self,input_dim, output_dim, width, depth, input_min, input_max, spatial_frequencies=(), l_ref=1.0, cross_ref=1.0):
        super().__init__()
        if input_dim != 4 or output_dim != 4:
            raise ValueError("Expected [X, Y, Z, s] inputs and four raw [ax, ay, az, p] outputs")
        self.coordinate_lengths = (float(l_ref), float(cross_ref))
        if not all(math.isfinite(length) and length > 0 for length in self.coordinate_lengths):
            raise ValueError("Reference lengths must be finite and positive")
        self.crossflow_ratio = self.coordinate_lengths[0] / self.coordinate_lengths[1]
        self.register_buffer("input_min", input_min, persistent=False)
        self.register_buffer("input_range", (input_max-input_min).clamp_min(1e-6), persistent=False)
        self.spatial_frequencies = tuple(float(f) for f in spatial_frequencies)
        self.register_buffer("fourier_frequencies", input_min.new_tensor(self.spatial_frequencies), persistent=False)
        self.input_layer = nn.Linear(input_dim + 6*len(self.spatial_frequencies), width)
        self.hidden_layers = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 1))
        self.output_layer = nn.Linear(width, output_dim)
        if self.spatial_frequencies:
            for layer in [self.input_layer, *self.hidden_layers]:
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("tanh"))
                nn.init.zeros_(layer.bias)
            # A spatial derivative multiplies each Fourier feature by its frequency.
            # Temper the initial slopes while keeping all frequencies trainable.
            with torch.no_grad():
                self.input_layer.weight[:, input_dim:].div_(
                    self.fourier_frequencies.abs().clamp_min(1.0).repeat(6)
                )
        # Zero potential slopes give the freestream velocity at initialization.
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return normalized [u, v, w, p] using the physically scaled curl."""
        features = 2.0 * (inputs - self.input_min) / self.input_range - 1.0
        if self.spatial_frequencies:
            frequencies = torch.pi * self.fourier_frequencies
            phase = features[:, :3, None] * frequencies
            sin, cos = phase.sin(), phase.cos()
            features = torch.cat((features, sin.flatten(1), cos.flatten(1)), dim=1)

        layers = (self.input_layer, *self.hidden_layers)
        activations = []
        for layer in layers:
            features = layer(features).tanh()
            activations.append(features)
        pressure = F.linear(features, self.output_layer.weight[3:], self.output_layer.bias[3:])

        # Propagate all three potential derivatives through the shared tanh layers.
        # These tensor operations stay differentiable for the momentum residual.
        jacobian = self.output_layer.weight[None, :3, :]
        for layer, activation in zip(reversed(layers), reversed(activations)):
            slope = 1.0 - activation.square()
            jacobian = (jacobian * slope.unsqueeze(1)) @ layer.weight

        spatial = jacobian[:, :, :3]
        if self.spatial_frequencies:
            feature_start = self.input_min.numel()
            d_sin, d_cos = jacobian[:, :, feature_start:].chunk(2, dim=-1)
            shape = (inputs.shape[0], 3, 3, len(self.spatial_frequencies))
            fourier_slopes = d_sin.reshape(shape) * cos[:, None] - d_cos.reshape(shape) * sin[:, None]
            spatial = spatial + (fourier_slopes * frequencies).sum(dim=-1)
        ax, ay, az = (spatial * (2.0 / self.input_range[:3])).unbind(dim=1)
        beta = self.crossflow_ratio
        velocity = torch.stack((
            1.0 + beta * (az[:, 1] - ay[:, 2]),
            beta * ax[:, 2] - az[:, 0],
            ay[:, 0] - beta * ax[:, 1],
        ), dim=1)
        return torch.cat((velocity, pressure), dim=1)
