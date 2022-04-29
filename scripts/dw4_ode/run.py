import sys
import os
import torch
sys.path.append(os.path.abspath("en_flows"))
from en_flows.dw4_experiment.dataset import get_data_dw4, remove_mean
from ffjord import FFJORD

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

    if torch.cuda.is_available():
        data_train = data_train.cuda()
        data_val = data_val.cuda()
        data_test = data_test.cuda()

    from sake.flow import SAKEDynamics, CenteredGaussian
    dynamics = SAKEDynamics(args.width, depth=args.depth)
    model = FFJORD(dynamics, trace_method="hutch", hutch_noise='bernoulli')
    x_prior = CenteredGaussian()

    if torch.cuda.is_available():
        x_prior = x_prior.cuda()
        model = model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)

    for idx_epoch in range(50000):
        x = data_train
        x = x - x.mean(dim=-2, keepdim=True)
        optimizer.zero_grad()
        z, delta_logp, reg_term = model(x)
        log_pz = x_prior.log_prob(z)
        log_px = (log_pz + delta_logp.view(-1)).mean()
        loss = -log_px
        loss.backward()
        optimizer.step()

        if idx_epoch % 100 == 0:
            x = data_val
            z, delta_logp, reg_term = model(x)
            log_pz = x_prior.log_prob(z)
            log_px = (log_pz + delta_logp.view(-1)).mean()
            loss_vl = -log_px

            x = data_test
            z, delta_logp, reg_term = model(x)
            log_pz = x_prior.log_prob(z)
            log_px = (log_pz + delta_logp.view(-1)).mean()
            loss_te = -log_px

            print(idx_epoch, loss, loss_vl, loss_te, flush=True)






if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--mp_depth", type=int, default=3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--cumulation", type=int, default=2)
    parser.add_argument("--n_data", type=int, default=100)
    args = parser.parse_args()
    run(args)
