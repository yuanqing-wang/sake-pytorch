import torch
import numpy as np
import sake

from torch.utils.data import Dataset, DataLoader

class _Dataset(Dataset):
    def __init__(self, ds):
        super().__init__()
        self.ds = ds
        self._prepare()

    def _prepare(self):
        idxs = np.cumsum(self.ds['N'])
        hs = np.split(self.ds['Z'], idxs)
        xs = np.split(self.ds['R'], idxs)
        fs = np.split(self.ds['F'], idxs)
        us = self.ds['E']
        self.hs = hs
        self.xs = xs
        self.fs = fs
        self.us = us

    def __len__(self):
        return len(self.us)

    def __getitem__(self, idxs):
        return torch.tensor(self.hs[idxs]), torch.tensor(self.xs[idxs]), torch.tensor(self.fs[idxs]), torch.tensor(self.us[idxs])


    def get_statistics(self):
        return self.fs.mean(), self.fs.std()


def padding(xs):
    hs, xs, fs, us = zip(*xs)
    _zs = [h for h in hs]
    masks = [torch.ones_like(z) for z in _zs]
    masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True)
    masks = masks.unsqueeze(-1) * masks.unsqueeze(-2)
    hs = [torch.nn.functional.one_hot(h, 10) for h in hs]
    hs = torch.nn.utils.rnn.pad_sequence(hs, batch_first=True)
    xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
    fs = torch.nn.utils.rnn.pad_sequence(fs, batch_first=True)
    us = torch.stack(us, dim=0).unsqueeze(-1)
    return hs, xs, us, fs, masks


def run(args):
    ds_tr = _Dataset(np.load("coll_v1.2_train.npz"))
    ds_vl = _Dataset(np.load("coll_v1.2_val.npz"))
    ds_te = _Dataset(np.load("coll_v1.2_test.npz"))
    ds_tr_loader = DataLoader(ds_tr, collate_fn=padding, batch_size=32, pin_memory=True)


    model = sake.DenseSAKEModel(
        in_features=10,
        hidden_features=args.width,
        depth=args.depth,
        out_features=args.width,
        update_coordinate=True,
        n_coefficients=32,
        distance_filter=sake.ContinuousFilterConvolution,
        activation=torch.nn.SiLU(),
    )

    from sake.utils import Readout
    readout = Readout(args.width, args.width, 1)

    if torch.cuda.is_available():
        model = model.cuda()
        readout = readout.cuda()

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(readout.parameters()), 
        args.lr
    )

    for _ in range(100):
        for h, x, u, f, mask in ds_tr_loader:
            h = h.float()
            x = x.float()
            u = u.float()
            f = f.float()
            mask = mask.float()

            x.requires_grad = True

            if torch.cuda.is_available():
                h = h.cuda()
                x = x.cuda()
                u = u.cuda()
                f = f.cuda()
                mask = mask.cuda()

            optimizer.zero_grad()
            u_hat, _ = model(h, x, mask=mask)
            u_hat = readout(u_hat, mask=mask.sum(dim=-1, keepdims=True).sign()) 
            f_hat = torch.autograd.grad(
                u_hat.sum(),
                x,
                create_graph=True,
            )[0]

            loss = torch.nn.MSELoss()(f_hat, f)
            loss.backward()
            print(loss.pow(0.5))
            optimizer.step()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--width", type=int, default=128)
    args = parser.parse_args()
    run(args)
