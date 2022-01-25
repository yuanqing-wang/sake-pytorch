import sys
import os
import torch
sys.path.append(os.path.abspath("en_flows"))
from en_flows.dw4_experiment.dataset import get_data_dw4, remove_mean

def run():
    import numpy as np

    data_train, batch_iter_train = get_data_dw4(100, 'train', 100)
    data_val, batch_iter_val = get_data_dw4(100, 'val', 100)
    data_test, batch_iter_test = get_data_dw4(100, 'test', 100)

    data_train = data_train.reshape(-1, 4, 2)
    data_val = data_val.reshape(-1, 4, 2)
    data_test = data_test.reshape(-1, 4, 2)

    data_train = data_train - data_train.mean(dim=-2, keepdim=True)
    data_val = data_val - data_val.mean(dim=-2, keepdim=True)
    data_test = data_test - data_test.mean(dim=-2, keepdim=True)

    from sake.flow import SAKEFlowModel, CenteredGaussian
    model = SAKEFlowModel(1, 32, depth=4)

    h = torch.zeros(100, 4, 1)
    x = data_train
    x_prior = CenteredGaussian()
    v_prior = CenteredGaussian()

    if torch.cuda.is_available():
        h = h.cuda()
        x = x.cuda()
        x_prior = x_prior.cuda()
        v_prior = v_prior.cuda()
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-12)

    for idx_epoch in range(20000):
        optimizer.zero_grad()
        v = v_prior.sample(x.shape)
        loss = model.nll_backward(h, x, v, x_prior, v_prior)
        loss.backward()
        optimizer.step()

        if idx_epoch % 100 == 0:
            x = data_val
            h = torch.zeros(x.shape[0], 4, 1)
            if torch.cuda.is_available():
                h = h.cuda()
                x = x.cuda()
            v = v_prior.sample(x.shape)
            loss_vl = model.nll_backward(h, x, v, x_prior, v_prior) + v_prior.log_prob(v).mean()

            x = data_test
            h = torch.zeros(x.shape[0], 4, 1)
            v = v_prior.sample(x.shape)
            if torch.cuda.is_available():
                h = h.cuda()
                x = x.cuda()
            loss_te = model.nll_backward(h, x, v, x_prior, v_prior) + v_prior.log_prob(v).mean()

            print(idx_epoch, "vl: %.4f, te: %.4f" % (loss_vl, loss_te))



if __name__ == "__main__":
    run()
