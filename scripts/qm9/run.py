import torch
import numpy as np
import dgl
import sake
from typing import Callable

from qm9 import dataset
from qm9 import utils as qm9_utils

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
        h = h * torch.sign(mask.sum(dim=-1, keepdim=True))
        h = h.sum(dim=-2)
        h = self.after_sum(h)
        return h

def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    dataloaders, charge_scale = dataset.retrieve_dataloaders(args.batch_size, 4)
    # compute mean and mean absolute deviation
    mu, sigma = qm9_utils.compute_mean_mad(dataloaders, args.property)
    coloring = sake.Coloring(mu, sigma)

    from sake.baselines.egnn import EGNNLayer
    model = sake.DenseSAKEModel(
        in_features=10,
        hidden_features=128,
        out_features=128,
        depth=7,
        update_coordinate=False,
        activation=torch.nn.SiLU(),
        layer=EGNNLayer,
    )
    readout = Readout(in_features=128, hidden_features=128, out_features=1)

    if torch.cuda.is_available():
        model = model.cuda()
        readout = readout.cuda()

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(readout.parameters()),
        args.lr
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    for idx_epoch in range(args.n_epochs):
        for data in dataloaders['train']:
            batch_size, n_nodes, _ = data['positions'].size()
            atom_positions = data['positions'].view(batch_size * n_nodes, -1).to(device, dtype)
            atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device, dtype)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)
            charges = data['charges'].to(device, dtype)
            nodes = qm9_utils.preprocess_input(one_hot, charges, args.charge_power, charge_scale, device)
            print(nodes.shape)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--property", type=str, default="U")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--charge_power", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=96)
    args = parser.parse_args()
    run(args)
