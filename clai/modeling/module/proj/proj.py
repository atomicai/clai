from typing import List

import torch


class ProjectionHead(torch.nn.Module):
    """A feed forward neural network of variable depth and width."""

    def __init__(self, layer_dims: List[int]) -> None:
        super(ProjectionHead, self).__init__()
        self.layer_dims = layer_dims
        layers = []

        for size_in, size_out in zip(layer_dims[:-1], layer_dims[1:]):
            layer = torch.nn.Linear(size_in, size_out)
            layers.append(layer)
        self.feed_forward = torch.nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        logits = self.feed_forward(input)
        return logits
