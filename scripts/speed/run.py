import os
import numpy as np
import torch
import sake
import networkx as nx 
import dgl

from torch.cuda.amp import autocast, GradScaler

def run(args):
    print(args)

    # os.mkdir(args.out)

    data = np.load("%s_dft.npz" % args.data)
    np.random.seed(2666)
    idxs = np.random.permutation(len(data['R']))

    x = data['R'][idxs]
    e = data['E'][idxs]
    i = data['z']# [idxs]
    f = data['F'][idxs]

    i = torch.nn.functional.one_hot(torch.tensor(i).type(torch.int64)).float()[None, :, :]
    x = torch.tensor(x).float()
    e = torch.tensor(e).float()
    f = torch.tensor(f).float()

    e_mean = e.mean()
    e_std = e.std()

    from functools import partial
    
    if "old" in args.data or "toluene" in args.data:
        in_features = 7

    else:
        in_features = 9


    model = sake.TandemDenseSAKEModel(
            in_features=in_features, 
            hidden_features=args.hidden_features,
            depth=args.depth,
            out_features=1, 
            # update_coordinate=True,
            n_coefficients=32,
            # distance_filter=sake.ContinuousFilterConvolution,
            activation=torch.nn.SiLU(),
    )

    n_tr = args.n_tr
    n_vl = args.n_vl
    batch_size = args.batch_size

    if n_vl == 0:
        n_vl = n_tr
    
    i = i.repeat(batch_size, 1, 1)

    x_tr = x[:n_tr]
    e_tr = e[:n_tr]
    f_tr = f[:n_tr]

    x_vl = x[n_tr:n_tr+n_vl]
    e_vl = e[n_tr:n_tr+n_vl]
    f_vl = f[n_tr:n_tr+n_vl]

    x_te = x[n_tr+n_vl:]
    e_te = e[n_tr+n_vl:]
    f_te = f[n_tr+n_vl:]

    n_te = len(x_te)
    coloring = sake.Coloring(e_mean, e_std)

    if torch.cuda.is_available():
        model = model.cuda()
        
        x_tr = x_tr.cuda()
        e_tr = e_tr.cuda()
        f_tr = f_tr.cuda()

        x_vl = x_vl.cuda()
        e_vl = e_vl.cuda()
        f_vl = f_vl.cuda()

        x_te = x_te.cuda()
        e_te = e_te.cuda()
        f_te = f_te.cuda()
        i = i.cuda()


    
    # model = torch.jit.trace(model, (i, x_tr[:batch_size]))
    # model = torch.jit.script(model)
    scaler = GradScaler()

    x_tr.requires_grad = True
    x_vl.requires_grad = True
    x_te.requires_grad = True
    optimizer = torch.optim.Adam(
            model.parameters(),
            args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.8, min_lr=1e-6)
    losses_vl = []

    for _ in range(5):
        idxs = torch.randperm(n_tr)
        for idx_batch in range(int(n_tr / batch_size)):
            _x_tr = x_tr[idxs[idx_batch*batch_size:(idx_batch+1)*batch_size]]
            _e_tr = e_tr[idxs[idx_batch*batch_size:(idx_batch+1)*batch_size]]
            _f_tr = f_tr[idxs[idx_batch*batch_size:(idx_batch+1)*batch_size]]

            optimizer.zero_grad()

            e_tr_pred, _ = model(i, _x_tr)
            e_tr_pred = e_tr_pred.sum(dim=1)
            e_tr_pred = coloring(e_tr_pred)

            f_tr_pred = -1.0 * torch.autograd.grad(
                e_tr_pred.sum(),
                _x_tr,
                create_graph=True,
            )[0]

            loss = torch.nn.MSELoss()(_f_tr, f_tr_pred)

            loss.backward()
            optimizer.step()


    import time
    time0 = time.time()
    model.train()
    idxs = torch.randperm(n_tr)
    for idx_batch in range(int(n_tr / batch_size)):
        _x_tr = x_tr[idxs[idx_batch*batch_size:(idx_batch+1)*batch_size]]
        _e_tr = e_tr[idxs[idx_batch*batch_size:(idx_batch+1)*batch_size]]
        _f_tr = f_tr[idxs[idx_batch*batch_size:(idx_batch+1)*batch_size]]

        optimizer.zero_grad()

        e_tr_pred, _ = model(i, _x_tr)
        e_tr_pred = e_tr_pred.sum(dim=1)
        e_tr_pred = coloring(e_tr_pred)

        f_tr_pred = -1.0 * torch.autograd.grad(
            e_tr_pred.sum(),
            _x_tr,
            create_graph=True,
        )[0]

        loss = torch.nn.MSELoss()(_f_tr, f_tr_pred)

        loss.backward()
        optimizer.step()

    time1 = time.time()
    avg_time = (time1 - time0) / float(idx_batch + 1)
    print(avg_time)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="malonaldehyde")
    parser.add_argument("--n_tr", type=int, default=1000)
    parser.add_argument("--n_vl", type=int, default=0)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--hidden_features", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--n_epoch", type=int, default=3000)
    parser.add_argument("--out", type=str, default="out")
    args = parser.parse_args()
    run(args)
