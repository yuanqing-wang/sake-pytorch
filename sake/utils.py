import torch
from typing import Callable
import numpy as np

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
        self.w_mu = torch.nn.Parameter(torch.ones(in_features, 1) * mu)
        self.w_sigma = torch.nn.Parameter(torch.ones(in_features, 1) * sigma)

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
            -(x-self.mu).pow(2) * self.gamma
        )

class HardCutOff(torch.nn.Module):
    def __init__(self, cutoff=5.0):
        super(HardCutOff, self).__init__()
        self.register_buffer("cutoff", torch.tensor(cutoff))

    def forward(self, x):
        return torch.where(
            torch.gt(x, 0.0) * torch.lt(x, self.cutoff),
            1.0,
            1e-14,
        )

class ContinuousFilterConvolutionWithConcatenation(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Callable=torch.nn.SiLU(),
        kernel: Callable=RBF(),
    ):
        super(ContinuousFilterConvolutionWithConcatenation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel = kernel
        kernel_dimension = kernel.out_features
        self.mlp_in = torch.nn.Linear(in_features, kernel_dimension)
        self.mlp_out = torch.nn.Sequential(
            torch.nn.Linear(kernel_dimension + 1, out_features),
            activation,
            torch.nn.Linear(out_features, out_features),
            activation,
        )

    def forward(self, h, x):
        h = self.mlp_in(h)
        _x = self.kernel(x)
        h = self.mlp_out(torch.cat([h * _x, x], dim=-1)) # * (1.0 - torch.eye(x.shape[-2], x.shape[-2], device=x.device).unsqueeze(-1))

        return h

class ContinuousFilterConvolutionWithConcatenationRecurrent(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Callable=torch.nn.SiLU(),
        kernel: Callable=RBF(),
        seq_dimension=1,
    ):
        super(ContinuousFilterConvolutionWithConcatenationRecurrent, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel = kernel
        self.seq_dimension = seq_dimension
        kernel_dimension = kernel.out_features
        self.kernel_dimension = kernel_dimension
        self.mlp_in = torch.nn.Linear(in_features, seq_dimension * kernel_dimension)
        self.mlp_out = torch.nn.Sequential(
            torch.nn.Linear(kernel_dimension * seq_dimension + seq_dimension, out_features),
            activation,
            torch.nn.Linear(out_features, out_features),
            activation,
        )


    def forward(self, h, x):
        # (batch_size, n, n, t * kernel_dimension)
        h = self.mlp_in(h)

        # (batch_size, t, n, n, kernel_dimension)
        h = h.view(
            *h.shape[:-4],
            self.seq_dimension,
            h.shape[-2],
            h.shape[-2],
            self.kernel_dimension
        )

        # (batch_size, t, n, n, kernel_dimension)
        _x = self.kernel(x)
        _x = h * _x

        # (batch_size, n, n, kernel_dimension * t)
        _x = _x.view(
            *_x.shape[:-4],
            _x.shape[-2],
            _x.shape[-2],
            self.seq_dimension * self.kernel_dimension
        )

        # (batch_size, n, n, t)
        x = x.movedim(-4, -1).flatten(-2, -1)

        # (batch_size, n, n, d)
        h = self.mlp_out(torch.cat([_x, x], dim=-1))
        return h


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
        h = self.mlp_out(h * x) # * (1.0 - torch.eye(x.shape[-2], x.shape[-2], device=x.device).unsqueeze(-1))

        return h

class ConcatenationFilter(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Callable=torch.nn.SiLU(),
    ):
        super(ConcatenationFilter, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features + 1, out_features),
            activation,
            torch.nn.Linear(out_features, out_features),
            activation,
        )

    def forward(self, h, x):
        h = self.mlp(
            torch.cat([h, x], dim=-1),
        )
        return h

class Readout(torch.nn.Module):
    def __init__(
            self,
            in_features:int=128,
            hidden_features:int=128,
            out_features:int=1,
            activation: Callable=torch.nn.SiLU(),
        ):
        super().__init__()
        self.before_sum = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, hidden_features),
        )
        self.after_sum = torch.nn.Sequential(
            torch.nn.Linear(hidden_features, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, out_features),
        )

    def forward(self, h, mask=None):
        h = self.before_sum(h)
        h = h * mask # torch.sign(mask.sum(dim=-1, keepdim=True))
        h = h.sum(dim=-2)
        h = self.after_sum(h)
        return h


def bootstrap(metric, n_samples=100, ci=0.95):
    def _bootstraped(input, target, metric=metric, n_samples=n_samples, ci=ci):
        original = metric(input=input, target=target).item()
        results = []
        for _ in range(n_samples):
            idxs = torch.multinomial(
                torch.ones(input.shape[0]),
                num_samples=n_samples,
                replacement=True,
            )

            _result = metric(
                input=input[idxs],
                target=target[idxs],
            ).item()

            results.append(_result)

        results = np.array(results)

        low = np.percentile(results, 100.0 * 0.5 * (1 - ci))
        high = np.percentile(results, (1 - ((1 - ci) * 0.5)) * 100.0)

        return original, low, high

    return _bootstraped

def assert_almost_equal_tensor(x0, x1, *args, **kwargs):
    import numpy.testing as npt
    npt.assert_almost_equal(
        x0.cpu().detach().numpy(),
        x1.cpu().detach().numpy(),
        *args, **kwargs,
    )
