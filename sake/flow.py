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

class VelocityDenseSAKEModelWithHistory(VelocityDenseSAKEModel):
    def forward(self, h, x):
        xs = [x]
        vs = []
        h = self.embedding_in(h)
        v = None
        for idx, eq_layer in enumerate(self.eq_layers):
            h, x, v = eq_layer(h, x, v)
            xs.append(x)
            vs.append(v)
        xs = torch.stack(xs, dim=-1)
        vs = torch.stack(vs, dim=-1)
        h = self.embedding_out(h)
        return h, xs, vs

class SAKEFlowLayer(HamiltonianFlowLayer):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        depth: int=4,
        activation: Callable=torch.nn.SiLU(),
        clip: bool=False,
    ):
        super().__init__()
        self.sake_model = VelocityDenseSAKEModelWithHistory(
            in_features=in_features+1,
            out_features=hidden_features,
            hidden_features=hidden_features,
            activation=activation,
            depth=depth,
            distance_filter=ContinuousFilterConvolutionWithConcatenation,
            update_coordinate=True,
            # tanh=True,
        )

        self.translation_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_features, hidden_features),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_features, 2*depth+1),
            torch.nn.Tanh(),
        )

        self.scale_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_features, hidden_features),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_features, 1),
            torch.nn.Tanh(),
        )

        self.clip = clip

    def mp(self, h, x):
        h = torch.cat([h, x.pow(2).sum(-1, keepdim=True)], dim=-1)
        h = torch.cat([h, torch.zeros_like(h[..., -1, :].unsqueeze(-2))], dim=-2)
        x = torch.cat([x, torch.zeros_like(x[..., -1, :]).unsqueeze(-2)], dim=-2)
        h, xs, vs = self.sake_model(h, x)
        xs = xs[..., :-1, :, :]
        vs = vs[..., :-1, :, :]
        h = h[..., :-1, :]

        translation = torch.cat([xs, vs], dim=-1)
        translation = translation - translation.mean(dim=-3, keepdim=True)
        translation_norm = translation.norm(dim=(-2, -3), keepdim=True)
        if self.clip:
            max_translation_norm = translation.shape[-2] * translation.shape[-3] * 1.0
            clipped_translation_norm = torch.clip(translation_norm, max=max_translation_norm)
            norm_scaling = clipped_translation_norm / (translation_norm + 1e-10)
        else:
            norm_scaling = 1.0 / (translation_norm + 1e-10)

        translation = translation * norm_scaling
        # translation = translation / (translation_norm + 1e-10)

        # (n_batch, n_atoms, 3)
        translation = (self.translation_mlp(h).unsqueeze(-2) * translation).sum(dim=-1)
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
            clip: bool=False,
            log_gamma: float=0.0,
        ):
        super().__init__()
        self.depth = depth
        self.embedding_in = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, hidden_features),
        )

        self.log_gamma = torch.nn.Parameter(torch.tensor(log_gamma))
        self.xv_layers = torch.nn.ModuleList()
        self.vx_layers = torch.nn.ModuleList()
        for _ in range(depth):
            # self.xv_layers.append(NonEquivariantSAKEFlowLayer())
            # self.vx_layers.append(NonEquivariantSAKEFlowLayer())


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
        x = x * self.log_gamma
        return x, v, sum_log_det

    def f_backward(self, h, x, v):
        h = self.embedding_in(h)
        sum_log_det = 0.0
        x = x * (-self.log_gamma).exp()
        sum_log_det = sum_log_det - self.log_gamma
        for xv_layer, vx_layer in zip(self.xv_layers[::-1], self.vx_layers[::-1]):
            v, x, log_det = vx_layer.f_backward(h, v, x)
            x, v = x - x.mean(dim=-2, keepdim=True), v - v.mean(dim=-2, keepdim=True)
            sum_log_det = sum_log_det + log_det

            x, v, log_det = xv_layer.f_backward(h, x, v)
            x, v = x - x.mean(dim=-2, keepdim=True), v - v.mean(dim=-2, keepdim=True)
            sum_log_det = sum_log_det + log_det
        return x, v, sum_log_det

    def nll_backward(self, h, x, v, x_prior, v_prior):
        x, v, sum_log_det = self.f_backward(h, x, v)
        nll_x = -x_prior.log_prob(x).mean()
        nll_v = -v_prior.log_prob(v).mean()
        return nll_x + nll_v + sum_log_det.mean()

class CenteredGaussian(torch.distributions.Normal):
    def __init__(self, scale=1.0):
        super().__init__(loc=0.0, scale=scale)
        self.device = "cpu"

    def to(self, device):
        self.loc = self.loc.to(device)
        self.scale = self.scale.to(device)
        self.device = device
        return self

    def cuda(self):
        return self.to("cuda:0")

    def cpu(self):
        return self.to("cpu")

    def log_prob(self, value):
        N = value.shape[-2]
        D = value.shape[-1]
        degrees_of_freedom = (N-1) * D
        r2 = value.pow(2).flatten(-2, -1).sum(dim=-1) / self.scale.pow(2)
        log_normalizing_constant = -0.5 * degrees_of_freedom * math.log(2*math.pi)
        log_px = -0.5 * r2 + log_normalizing_constant
        return log_px

    def sample(self, *args, **kwargs):
        x = super().sample(*args, **kwargs)
        x = x - x.mean(dim=-2, keepdim=True)
        x = x.to(self.device)
        return x

    def rsample(self, *args, **kwargs):
        x = super().rsample(*args, **kwargs)
        x = x - x.mean(dim=-2, keepdim=True)
        x = x.to(self.device)
        return x
