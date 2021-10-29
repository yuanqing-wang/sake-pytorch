import torch

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

class RBF(torch.nn.Module):
    def __init__(
            self,
            gamma=10.0,
            mu=torch.arange(0, 30, 1.0),
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
