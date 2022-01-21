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
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)

    for _ in range(10000):
        optimizer.zero_grad()
        v = v_prior.sample(x.shape)
        loss = model.nll_backward(h, x, v, x_prior, v_prior)
        print(loss.item(), (loss+v_prior.log_prob(v).mean()).item())
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    run()
