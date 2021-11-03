import torch
from typing import Callable

AGGREGATORS = {
    'sum': torch.sum,
    'mean': torch.mean,
    'max': lambda *args, **kwargs: torch.max(*args, **kwargs)[0],
    'min': lambda *args, **kwargs: torch.min(*args, **kwargs)[0],
    'var': torch.var,
}

class PNA(torch.nn.Module):
    def __init__(self, aggregators=['sum', 'mean', 'max', 'min', 'var']):
        super(PNA, self).__init__()
        self.aggregators = aggregators
        self.n_aggregators = len(self.aggregators)

    def forward(self, x, dim=1):
        x = torch.cat(
            [
                AGGREGATORS[aggregator](x, dim=dim)
                for aggregator in self.aggregators
            ],
            dim=-1
        )

        return x

class Coloring(torch.nn.Module):
    def __init__(
        self,
        mu=0.0,
        sigma=1.0,
    ):
        super(Coloring, self).__init__()
        self.register_buffer("mu", torch.tensor(mu))
        self.register_buffer("sigma", torch.tensor(sigma))

    def forward(self, x):
        return self.sigma * x + self.mu

class ConditionalColoring(torch.nn.Module):
    def __init__(
        self,
        in_features,
        mu=0.0,
        sigma=1.0,
    ):
        super(ConditionalColoring, self).__init__()
        self.register_buffer("w_mu", torch.ones(in_features, 1) * mu)
        self.register_buffer("w_sigma", torch.ones(in_features, 1) * sigma)

    def forward(self, i, x):
        mu = i @ self.w_mu
        sigma = i @ self.w_sigma
        return x * sigma + mu

class RBF(torch.nn.Module):
    def __init__(
            self,
            gamma=10.0,
            mu=torch.linspace(0, 30, 300),
            ):
        super(RBF, self).__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("mu", torch.tensor(mu))
        self.out_features = len(mu)

    def forward(self, x):
        return torch.exp(
            -(x-self.mu.view(*[1 for _ in range(x.dim()-1)], -1)).pow(2) * self.gamma
        )

class HardCutOff(torch.nn.Module):
    def __init__(self, cutoff=5.0):
        super(HardCutOff, self).__init__()
        self.register_buffer("cutoff", torch.tensor(cutoff))

    def forward(self, x):
        return torch.where(
            torch.gt(x, 0.0) * torch.lt(x, self.cutoff),
            1.0,
            0.0,
        )

class ContinuousFilterConvolution(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Callable=torch.nn.SiLU(),
        kernel: Callable=RBF(),
    ):
        super(ContinuousFilterConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel = kernel
        kernel_dimension = kernel.out_features
        self.mlp_in = torch.nn.Linear(in_features, kernel_dimension)
        self.mlp_out = torch.nn.Sequential(
            torch.nn.Linear(kernel_dimension, out_features),
            activation,
        )

    def forward(self, h, x):
        h = self.mlp_in(h)
        x = self.kernel(x)        
        h = self.mlp_out(h * x)  * (1.0 - torch.eye(x.shape[-2], x.shape[-2], device=x.device).unsqueeze(-1))

        return h
