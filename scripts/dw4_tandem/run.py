import sys
import os
import torch
sys.path.append(os.path.abspath("en_flows"))
from en_flows.dw4_experiment.dataset import get_data_dw4, remove_mean

def run(args):
    print(args)
    import numpy as np

    data_train, batch_iter_train = get_data_dw4(args.n_data, 'train', 100)
    data_val, batch_iter_val = get_data_dw4(args.n_data, 'val', 100)
    data_test, batch_iter_test = get_data_dw4(args.n_data, 'test', 100)

    data_train = data_train.reshape(-1, 4, 2)
    data_val = data_val.reshape(-1, 4, 2)
    data_test = data_test.reshape(-1, 4, 2)

    data_train = data_train - data_train.mean(dim=-2, keepdim=True)
    data_val = data_val - data_val.mean(dim=-2, keepdim=True)
    data_test = data_test - data_test.mean(dim=-2, keepdim=True)

    from sake.flow import SAKEFlowModel, CenteredGaussian
    model = SAKEFlowModel(
            1, args.width, depth=args.depth, mp_depth=args.mp_depth,
            log_gamma=data_train.norm(dim=(-1, -2), keepdim=True).mean().log()
    )
    
    print(model)

    v_prior = x_prior = CenteredGaussian()

    if torch.cuda.is_available():
        x_prior = v_prior = x_prior.cuda()
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.lr/args.cumulation, weight_decay=args.weight_decay)

    min_loss_vl = 1000.0
    min_loss_te = 1000.0

    for idx_epoch in range(50000):
        x = data_train
        x = x - x.mean(dim=-2, keepdim=True)
        h = torch.zeros(x.shape[0], 4, 1)
        if torch.cuda.is_available():
            x = x.cuda()
            h = h.cuda()
        optimizer.zero_grad()
        for _ in range(args.cumulation):
            v = x[torch.randperm(x.shape[0])]
            loss = model.nll_backward(h, x, v, x_prior, v_prior)
            loss.backward()
        optimizer.step()

        if idx_epoch % 1000 == 0:
            loss_vl = 0.0
            loss_te = 0.0
            for idx in range(10):
                x = data_train
                h = torch.zeros(x.shape[0], 4, 1)
                if torch.cuda.is_available():
                    h = h.cuda()
                    x = x.cuda()
                v = x[torch.randperm(x.shape[0])]
                loss_tr = 0.5 * model.nll_backward(h, x, v, x_prior, v_prior).item()

                x = data_val[100*idx:100*idx+100]
                h = torch.zeros(x.shape[0], 4, 1)
                if torch.cuda.is_available():
                    h = h.cuda()
                    x = x.cuda()
                v = x[torch.randperm(x.shape[0])]
                loss_vl += 0.5 * model.nll_backward(h, x, v, x_prior, v_prior).item()

                x = data_test[100*idx:100*idx+100]
                h = torch.zeros(x.shape[0], 4, 1)
                if torch.cuda.is_available():
                    h = h.cuda()
                    x = x.cuda()
                v = x[torch.randperm(x.shape[0])]
                loss_te += 0.5 * model.nll_backward(h, x, v, x_prior, v_prior).item()


            loss_vl *= 0.1
            loss_te *= 0.1
            print(idx_epoch, "tr: %.4f, vl: %.4f, te: %.4f" % (loss_tr, loss_vl, loss_te), flush=True)
        torch.save(model, "sake_flow_dw4.th")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--mp_depth", type=int, default=3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--cumulation", type=int, default=2)
    parser.add_argument("--n_data", type=int, default=100)
    args = parser.parse_args()
    run(args)
