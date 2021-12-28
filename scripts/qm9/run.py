import torch
import numpy as np
import dgl
import sake
from typing import Callable

def padding(xs):
    gs, ys = zip(*xs)
    _zs = [g.ndata['Z'] for g in gs]
    masks = [torch.ones_like(z) for z in _zs]
    masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True)
    masks = masks.unsqueeze(-1) * masks.unsqueeze(-2)
    zs = [torch.nn.functional.one_hot(z, 10) for z in _zs]
    rs = [g.ndata['R'] for g in gs]
    zs = torch.nn.utils.rnn.pad_sequence(zs, batch_first=True)
    rs = torch.nn.utils.rnn.pad_sequence(rs, batch_first=True)
    ys = torch.stack(ys, dim=0)
    return zs.float(), rs.float(), ys.float(), masks.float()

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
    from dgl.data import QM9Dataset
    data = QM9Dataset(label_keys=[args.key])

    frac_list = [100000, 18000, 13000]
    frac_list = [float(x) / sum(frac_list) for x in frac_list]

    ds_tr, ds_vl, ds_te = dgl.data.utils.split_dataset(data, random_state=0, shuffle=True, frac_list=frac_list)

    mu, sigma = data.label.mean(), data.label.std()
    coloring = sake.Coloring(mu, sigma)

    from torch.utils.data import DataLoader
    ds_tr_loader = DataLoader(ds_tr, batch_size=96, collate_fn=padding, pin_memory=True)
    ds_vl_loader = DataLoader(ds_vl, batch_size=96, collate_fn=padding, pin_memory=True)

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
        for z, r, y, mask in ds_tr_loader:
            if torch.cuda.is_available():
                z = z.cuda()
                r = r.cuda()
                y = y.cuda()
                mask = mask.cuda()

            optimizer.zero_grad()
            y_hat, _ = model(z, r, mask=mask)
            y_hat = readout(y_hat, mask=mask)
            y_hat = coloring(y_hat)
            loss = torch.nn.MSELoss()(y_hat, y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        with torch.no_grad():
            model.eval()
            ys = []
            ys_hat = []

            for z, r, y, mask in ds_vl_loader:
                z = z.cuda().float()
                r = r.cuda()
                y = y.cuda()
                mask = mask.cuda().float()

                y_hat, _ = model(z, r, mask=mask)
                y_hat = readout(y_hat, mask=mask)
                y_hat = coloring(y_hat)

                ys.append(y)
                ys_hat.append(y_hat)

            ys = torch.cat(ys, dim=0)
            ys_hat = torch.cat(ys_hat, dim=0)
            loss = torch.nn.L1Loss()(ys, ys_hat)
            print(loss, flush=True)
                


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str, default="U")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_epochs", type=int, default=1000)
    args = parser.parse_args()
    run(args)
