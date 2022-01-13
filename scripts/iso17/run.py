import os
import numpy as np
import torch
import sake
import networkx as nx 
import dgl
from ase.db import connect

def run(args):
    from torch.utils.data import Dataset, DataLoader
    class _Dataset(Dataset):
        def __init__(self, db=None, xs=None, hs=None, us=None, fs=None):
            super(_Dataset, self).__init__()
            if db is not None:
                self.db = db
                self._process()
            else:
                self.xs = xs
                self.hs = hs
                self.us = us
                self.fs = fs

        def _process(self):
            if self.db is not None:
                import numpy as np
                xs = []
                hs = []
                us = []
                fs = []
                with connect(self.db) as conn:
                    for row in conn.select():
                        xs.append(row['positions'])
                        us.append(row['total_energy'])

                        h = np.zeros((19, 10))
                        h[np.arange(19), row['numbers']] = 1
                        hs.append(h)

                        f = np.array(row.data['atomic_forces'])
                        fs.append(f)

                import numpy as np
                xs = np.stack(xs, axis=0)
                hs = np.stack(hs, axis=0)
                us = np.stack(us, axis=0)
                fs = np.stack(fs, axis=0)

                xs = xs.astype(np.float32)
                hs = hs.astype(np.float32)
                us = us.astype(np.float32)
                fs = fs.astype(np.float32)

                self.xs = xs
                self.hs = hs
                self.us = us
                self.fs = fs

        def __len__(self):
            return len(self.us)

        def __getitem__(self, idxs):
            return self.xs[idxs], self.hs[idxs], self.us[idxs], self.fs[idxs]

        def shuffle(self, seed=None):
            import random
            if seed is not None:
                random.seed(seed)
            idxs = list(range(len(self)))
            random.shuffle(idxs)
            self.xs = self.xs[idxs]
            self.us = self.us[idxs]
            self.hs = self.hs[idxs]
            self.fs = self.fs[idxs]
            return self

    ds = _Dataset("iso17/reference.db")
    idxs_tr = open("iso17/train_ids.txt").readlines()
    idxs_vl = open("iso17/validation_ids.txt").readlines()
    idxs_tr = [int(x.strip())-1 for x in idxs_tr]
    idxs_vl = [int(x.strip())-1 for x in idxs_vl]


    _ds_te = _Dataset("iso17/test_other.db")
    ds_te = DataLoader(_ds_te, batch_size=64, pin_memory=True)

    import random
    random.shuffle(idxs_tr)
    idxs_tr = idxs_tr[:5000]


    x, h, u, f = ds[idxs_tr]
    print(f.shape)
    _ds_tr = _Dataset(xs=x, hs=h, us=u, fs=f)
    x, h, u, f = ds[idxs_vl]
    _ds_vl = _Dataset(xs=x, hs=h, us=u, fs=f)
    ds_tr = DataLoader(_ds_tr, batch_size=64, shuffle=True, pin_memory=True)
    ds_vl = DataLoader(_ds_vl, batch_size=64, pin_memory=True)

    model = sake.DenseSAKEModel(
        in_features=10,
        hidden_features=64,
        out_features=1,
        update_coordinate=False, # [False, False, True, True],
        n_coefficients=8,
        distance_filter=sake.ContinuousFilterConvolution,
        depth=4,
        activation=torch.nn.SiLU(),
    )

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=100, min_lr=1e-6)
    
    for idx_epoch in range(100000):
        if idx_epoch % 50 == 0:
            print(idx_epoch)
            for x, h, u, f in ds_tr:
                x.requires_grad = True
                x = x.cuda()
                h = h.cuda()
                u = u.cuda()
                f = f.cuda()
                optimizer.zero_grad()
                h_hat, _ = model(h, x)
                u_hat = h_hat.sum(dim=(-2))

                f_hat = -1.0 * torch.autograd.grad(
                    u_hat,
                    x,
                    grad_outputs=torch.ones_like(u_hat),
                    retain_graph=True,
                    create_graph=True,
                )[0]

                loss = torch.nn.MSELoss()(f_hat, f)
                loss.backward()
                optimizer.step()

            model.eval()
            fs = []
            fs_hat = []
            for x, h, u, f in ds_vl:
                x.requires_grad = True
                x = x.cuda()
                h = h.cuda()
                u = u.cuda()
                f = f.cuda()
                h_hat, _ = model(h, x)
                u_hat = h_hat.sum(dim=(-2))

                f_hat = -1.0 * torch.autograd.grad(
                    u_hat,
                    x,
                    grad_outputs=torch.ones_like(u_hat),
                    retain_graph=True,
                    create_graph=True,
                )[0]


                fs.append(f.detach())
                fs_hat.append(f_hat.detach())

            fs = torch.cat(fs, dim=0)
            fs_hat = torch.cat(fs_hat, dim=0)

            loss_vl = torch.nn.L1Loss()(f_hat, f)
            scheduler.step(loss_vl)


            model.eval()
            fs = []
            fs_hat = []
            for x, h, u, f in ds_te:
                x.requires_grad = True
                x = x.cuda()
                h = h.cuda()
                u = u.cuda()
                f = f.cuda()
                h_hat, _ = model(h, x)
                u_hat = h_hat.sum(dim=(-2))

                f_hat = -1.0 * torch.autograd.grad(
                    u_hat,
                    x,
                    grad_outputs=torch.ones_like(u_hat),
                    retain_graph=True,
                    create_graph=True,
                )[0]


                fs.append(f.detach())
                fs_hat.append(f_hat.detach())

            fs = torch.cat(fs, dim=0)
            fs_hat = torch.cat(fs_hat, dim=0)

            loss_te = torch.nn.L1Loss()(f_hat, f)
            
            print(loss_vl, loss_te, flush=True)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    run(args)
