import torch
import numpy as np
import dgl
import sake
from typing import Callable

from qm9 import dataset
from qm9 import utils as qm9_utils

from torch.cuda.amp import autocast, GradScaler

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

def run(args):
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    dataloaders, charge_scale = dataset.retrieve_dataloaders(args.batch_size, 0)
    # compute mean and mean absolute deviation
    mu, sigma = qm9_utils.compute_mean_mad(dataloaders, args.property)
    coloring = sake.Coloring(mu, sigma)

    from sake.baselines.egnn import EGNNLayer
    from sake.layers import DenseSAKELayer
    model = sake.TandemDenseSAKEModel(
        in_features=15,
        hidden_features=args.width,
        out_features=args.width,
        depth=args.depth,
        activation=torch.nn.SiLU(),
    )
    readout = Readout(in_features=args.width, hidden_features=args.width, out_features=1)

    if torch.cuda.is_available():
        model = model.cuda()
        readout = readout.cuda()

    # model = torch.jit.script(model)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(readout.parameters()),
        args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
    scaler = GradScaler()

    for idx_epoch in range(args.n_epochs):
        scheduler.step()
        for data in dataloaders['train']:
            optimizer.zero_grad()
            batch_size, n_nodes, _ = data['positions'].size()
            atom_positions = data['positions'].view(batch_size, n_nodes, -1).to(device, dtype)
            atom_mask = data['atom_mask'].view(batch_size, n_nodes, -1).to(device, dtype)
            edge_mask = data['edge_mask'].to(device, dtype).view(batch_size, n_nodes, n_nodes)
            one_hot = data['one_hot'].to(device, dtype)
            charges = data['charges'].to(device, dtype)
            nodes = qm9_utils.preprocess_input(one_hot, charges, args.charge_power, charge_scale, device)

            with autocast():
                y_hat, _ = model(nodes, atom_positions, mask=edge_mask)
                y_hat = readout(y_hat, atom_mask)
                y = data[args.property].to(device, dtype).unsqueeze(-1)
                y_hat = coloring(y_hat)
                loss = torch.nn.L1Loss()(y_hat, y)
            # loss.backward()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()

        losses = []
        with torch.no_grad():
            ys = []
            ys_hat = []
            for data in dataloaders['valid']:
                batch_size, n_nodes, _ = data['positions'].size()
                atom_positions = data['positions'].view(batch_size, n_nodes, -1).to(device, dtype)
                atom_mask = data['atom_mask'].view(batch_size, n_nodes, -1).to(device, dtype)
                edge_mask = data['edge_mask'].to(device, dtype).view(batch_size, n_nodes, n_nodes)
                one_hot = data['one_hot'].to(device, dtype)
                charges = data['charges'].to(device, dtype)
                nodes = qm9_utils.preprocess_input(one_hot, charges, args.charge_power, charge_scale, device)
                y_hat, _ = model(nodes, atom_positions, mask=edge_mask)
                y_hat = readout(y_hat, atom_mask)
                y = data[args.property].to(device, dtype).unsqueeze(-1)
                y_hat = coloring(y_hat)

                ys.append(y)
                ys_hat.append(y_hat)

            ys = torch.cat(ys, dim=0)
            ys_hat = torch.cat(ys_hat, dim=0)
            loss = torch.nn.L1Loss()(ys, ys_hat)
            losses.append(loss.item())
        print(min(losses))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--property", type=str, default="homo")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--charge_power", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--weight_decay", type=float, default=1e-14)
    args = parser.parse_args()
    run(args)
