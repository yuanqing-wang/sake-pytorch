import abc
import torch
from .layers import DenseSAKELayer
from .models import VelocityDenseSAKEModel
from .functional import (
    get_x_minus_xt,
    get_x_minus_xt_norm,
    get_h_cat_h,
)
import math
from typing import Union, Callable
from .utils import ContinuousFilterConvolutionWithConcatenation

class HamiltonianFlowLayer(torch.nn.Module, abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @abc.abstractmethod
    def f_forward(self, x, v, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def f_backward(self, x, v, *args, **kwargs):
        raise NotImplementedError

class HamiltonianFlowModel(torch.nn.Module, abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @abc.abstractmethod
    def f_forward(self, x, v, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def f_backward(self, x, v, *args, **kwargs):
        raise NotImplementedError

    #
    # @abc.abstractmethod
    # def loss(self, z, *args, **kwargs):
    #     raise NotImplementedError

# class VelocityDenseSAKEModelWithHistory(VelocityDenseSAKEModel):
#     def forward(self, h, x):
#         xs = [x]
#         vs = []
#         h = self.embedding_in(h)
#         v = None
#         for idx, eq_layer in enumerate(self.eq_layers):
#             h, x, v = eq_layer(h, x, v)
#             xs.append(x)
#             vs.append(v)
#         xs = torch.stack(xs, dim=-1)
#         vs = torch.stack(vs, dim=-1)
#         h = self.embedding_out(h)
#         return h, xs, vs

class SAKEFlowLayer(HamiltonianFlowLayer):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        depth: int=4,
        activation: Callable=torch.nn.SiLU(),
        clip: bool=True,
    ):
        super().__init__()
        self.sake_model = VelocityDenseSAKEModel(
            in_features=in_features+1,
            out_features=hidden_features,
            hidden_features=hidden_features,
            activation=activation,
            depth=depth,
            distance_filter=ContinuousFilterConvolutionWithConcatenation,
            update_coordinate=True,
        )

        self.scale_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_features, hidden_features),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_features, 1),
            torch.nn.Tanh(),
        )

    def mp(self, h, x):
        x0 = x
        h = torch.cat([h, x.pow(2).sum(-1, keepdim=True)], dim=-1)
        h = torch.cat([h, torch.zeros_like(h[..., -1, :].unsqueeze(-2))], dim=-2)
        x = torch.cat([x, torch.zeros_like(x[..., -1, :]).unsqueeze(-2)], dim=-2)
        h, x = self.sake_model(h, x)
        x = x[..., :-1, :]
        h = h[..., :-1, :]

        translation = x - x0
        translation = translation - translation.mean(dim=-2, keepdim=True)
        scale = self.scale_mlp(h).mean(dim=-2, keepdim=True)
        return scale, translation

    def f_forward(self, h, x, v):
        scale, translation = self.mp(h, x)
        v = scale.exp() * v + translation
        log_det = scale.sum((-1, -2)) * v.shape[-1] * v.shape[-2]
        return x, v, log_det

    def f_backward(self, h, x, v):
        scale, translation = self.mp(h, x)
        v = v - translation
        v = (-scale).exp() * v
        log_det = scale.sum((-1, -2)) * v.shape[-1] * v.shape[-2]
        return x, v, log_det

class SAKEFlowModel(HamiltonianFlowModel):
    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            depth: int=4,
            mp_depth: int=4,
            activation: Callable=torch.nn.SiLU(),
            clip: bool=True,
            beta: float=1.0,
        ):
        super().__init__()
        self.depth = depth
        self.embedding_in = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, hidden_features),
        )


        self.xv_layers = torch.nn.ModuleList()
        self.vx_layers = torch.nn.ModuleList()

        for _ in range(depth):

            self.xv_layers.append(
                SAKEFlowLayer(
                    in_features=hidden_features,
                    hidden_features=hidden_features,
                    depth=mp_depth,
                    clip=clip,
                )
            )

            self.vx_layers.append(
                SAKEFlowLayer(
                    in_features=hidden_features,
                    hidden_features=hidden_features,
                    depth=mp_depth,
                    clip=clip,
                )
            )

        self.beta = beta

    def f_forward(self, h, x, v):
        h = self.embedding_in(h)
        sum_log_det = 0.0
        for xv_layer, vx_layer in zip(self.xv_layers, self.vx_layers):
            x, v, log_det = xv_layer.f_forward(h, x, v)
            x, v = x - x.mean(dim=-2, keepdim=True), v - v.mean(dim=-2, keepdim=True)
            sum_log_det = sum_log_det + log_det

            v, x, log_det = vx_layer.f_forward(h, v, x)
            x, v = x - x.mean(dim=-2, keepdim=True), v - v.mean(dim=-2, keepdim=True)
            sum_log_det = sum_log_det + log_det
        return x, v, sum_log_det

    def f_backward(self, h, x, v):
        h = self.embedding_in(h)
        sum_log_det_x = 0.0
        sum_log_det_v = 0.0
        for xv_layer, vx_layer in zip(self.xv_layers[::-1], self.vx_layers[::-1]):
            v, x, log_det = vx_layer.f_backward(h, v, x)
            x, v = x - x.mean(dim=-2, keepdim=True), v - v.mean(dim=-2, keepdim=True)
            sum_log_det_v = sum_log_det_v + log_det

            x, v, log_det = xv_layer.f_backward(h, x, v)
            x, v = x - x.mean(dim=-2, keepdim=True), v - v.mean(dim=-2, keepdim=True)
            sum_log_det_x = sum_log_det_x + log_det
        return x, v, sum_log_det_x, sum_log_det_v

    def nll_backward(self, h, x, v, x_prior, v_prior, beta=1.0):
        x, v, sum_log_det_x, sum_log_det_v = self.f_backward(h, x, v)
        nll_x = -x_prior.log_prob(x).mean()
        nll_v = -v_prior.log_prob(v).mean()
        return nll_x + beta * nll_v + sum_log_det_x.mean() + beta * sum_log_det_v.mean()


class HierarchicalSAKEFlowModel(HamiltonianFlowModel):
    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            depth: int=3,
            model_depth: int=3,
            mp_depth: int=3,
            activation: Callable=torch.nn.SiLU(),
            clip: bool=True,
            beta: float=1.0,
        ):
        
        super().__init__()
        self.depth = depth
        self.models = torch.nn.ModuleList()

        for _ in range(depth):
            self.models.append(
                SAKEFlowModel(
                    in_features=in_features,
                    hidden_features=hidden_features,
                    depth=model_depth,
                    mp_depth=mp_depth,
                    activation=activation,
                    clip=clip,
                    beta=beta,
                )
            )

        self.beta = beta

    def f_forward(self, h, shape):
        sum_log_det = 0.0
        x = self.prior.sample(shape)
        aux = []
        for model in self.models:
            v = self.prior.sample(shape)
            x, v, log_det = model.f_forward(h, x, v)
            aux.append(v)
            aux.append(x)
            sum_log_det = sum_log_det + log_det
        aux.pop()
        return x, sum_log_det, aux

    def f_backward(self, h, x):
        sum_log_det = 0.0
        shape = x.shape
        aux = []
        for model in self.models[::-1]:
            v = self.prior.sample(shape)
            x, v, log_det = model.f_backward(h, x, v)
            aux.append(v)
            aux.append(x)
            sum_log_det = sum_log_det + log_det
        return x, sum_log_det, aux

    def nll_backward(self, h, x):
        x, sum_log_det, aux = self.f_backward(h, x)
        aux = torch.stack(aux, dim=0)
        nll = -self.prior.log_prob(aux).sum(dim=0).mean()
        return nll + sum_log_det.mean()

class SAKEDynamics(torch.nn.Module):
    def __init__(
            self,
            hidden_features: int,
            depth: int,
            activation: Callable=torch.nn.SiLU(),
        ):
        super().__init__()

        self.embedding_in = torch.nn.Sequential(
            torch.nn.Linear(2, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, hidden_features),
        )

        self.sake_model = VelocityDenseSAKEModel(
            in_features=hidden_features,
            out_features=hidden_features,
            hidden_features=hidden_features,
            activation=activation,
            depth=depth,
            distance_filter=ContinuousFilterConvolutionWithConcatenation,
            update_coordinate=True,
        )

    def forward(self, t, x):
        t = t * torch.ones(x.shape[:-1], device=x.device).unsqueeze(-1)
        h = torch.cat([t, x.pow(2).sum(-1, keepdim=True)], dim=-1)
        h = self.embedding_in(h)
        h, x1 = self.sake_model(h, x)
        x = x1 - x
        x = x - x.mean(dim=-2, keepdim=True)
        return x






class CenteredGaussian(torch.nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.register_buffer("loc", torch.tensor(0.0))
        self.register_buffer("scale", torch.tensor(scale))

    def log_prob(self, value):
        N = value.shape[-2]
        D = value.shape[-1]
        degrees_of_freedom = (N-1) * D
        r2 = value.pow(2).flatten(-2, -1).sum(dim=-1) / self.scale.pow(2)
        log_normalizing_constant = -0.5 * degrees_of_freedom * math.log(2*math.pi)
        log_px = -0.5 * r2 + log_normalizing_constant
        return log_px

    def sample(self, *args, **kwargs):
        x = torch.distributions.Normal(loc=self.loc, scale=self.scale).sample(*args, **kwargs)
        x = x - x.mean(dim=-2, keepdim=True)
        x = x.to(self.device)
        return x

    def rsample(self, *args, **kwargs):
        x = torch.distributions.Normal(loc=self.loc, scale=self.scale).rsample(*args, **kwargs)
        x = x - x.mean(dim=-2, keepdim=True)
        x = x.to(self.device)
        return x
