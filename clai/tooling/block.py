import torch.nn as nn


class UpscaleBlock(nn.Module):
    def __init__(self, x_dim, h_dim):
        super(UpscaleBlock, self).__init__()
        self.xh = FeedForwardBlock(layer_dims=[x_dim, h_dim])
        self.f = nn.SiLU()
        self.hx = FeedForwardBlock(layer_dims=[h_dim, x_dim])

    def forward(self, X):
        y = self.xh(X)
        y = self.f(y)
        y = self.hx(y)
        return y


class FeedForwardBlock(nn.Module):
    """A feed forward neural network of variable depth and width."""

    def __init__(self, layer_dims, **kwargs):
        # Todo: Consider having just one input argument
        super(FeedForwardBlock, self).__init__()
        self.layer_dims = layer_dims
        # If read from config the input will be string
        n_layers = len(layer_dims) - 1
        layers_all = []

        for i in range(n_layers):
            size_in = layer_dims[i]
            size_out = layer_dims[i + 1]
            layer = nn.Linear(size_in, size_out)
            layers_all.append(layer)
        self.feed_forward = nn.Sequential(*layers_all)

    def forward(self, X):
        logits = self.feed_forward(X)
        return logits
