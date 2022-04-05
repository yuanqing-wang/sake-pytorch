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
            mu=torch.linspace(0, 30, 300),
            gamma=10.0,
            ):
        super(RBF, self).__init__()

        if gamma is None:
            gamma = 0.5 / (mu[1] - mu[0]) ** 2

        self.gamma = torch.nn.Parameter(torch.tensor(gamma))
        self.mu = torch.nn.Parameter(torch.tensor(mu))
        # self.register_buffer("gamma", torch.tensor(gamma))
        # self.register_buffer("mu", torch.tensor(mu))
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
        kernel: Callable=RBF,
    ):
        super(ContinuousFilterConvolutionWithConcatenation, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel = kernel()
        kernel_dimension = self.kernel.out_features
        self.mlp_in = torch.nn.Linear(in_features, kernel_dimension)
        self.mlp_out = torch.nn.Sequential(
            torch.nn.Linear(in_features + kernel_dimension + 1, out_features),
            activation,
            torch.nn.Linear(out_features, out_features),
        )


    def forward(self, h, x):
        h0 = h
        h = self.mlp_in(h)
        _x = self.kernel(x) * h
        h = self.mlp_out(torch.cat([h0, _x, x], dim=-1)) # * (1.0 - torch.eye(x.shape[-2], x.shape[-2], device=x.device).unsqueeze(-1))
        return h


class ExpNormalSmearing(torch.nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=True):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", torch.nn.Parameter(means))
            self.register_parameter("betas", torch.nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(
            torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower)
        )
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        return torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )

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
            *h.shape[:-3],
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
