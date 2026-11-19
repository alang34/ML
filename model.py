import torch
from torch import nn

class PINN(nn.Module):
    def __init__(self,input_dim, output_dim, width, depth, input_min, input_max, spatial_frequencies=()):
        super().__init__()
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
            nn.init.xavier_uniform_(self.output_layer.weight, gain=0.1)
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = 2.0*(inputs-self.input_min)/self.input_range-1.0
        if self.spatial_frequencies:
            phase = torch.pi*inputs[:, :3, None]*self.fourier_frequencies
            inputs = torch.cat([inputs, phase.sin().flatten(1), phase.cos().flatten(1)], dim=1)
        values = torch.tanh(self.input_layer(inputs))
        for layer in self.hidden_layers:
            values = torch.tanh(layer(values))
        return self.output_layer(values)
