import copy
import torch
import gc


def fed_avg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))

    return w_avg


def fed_avg_saveClient(weight_dir, model_type, num_users):
    w_avg = (torch.load(weight_dir + model_type + 'client{:d}.pth'.format(1))).state_dict()
    for k in w_avg.keys():
        for i in range(1, num_users):
            w = torch.load(weight_dir + model_type + 'client{:d}.pth'.format(i + 1)).state_dict()
            w_avg[k] += w[k]
            del w
            gc.collect()
        w_avg[k] = torch.div(w_avg[k], num_users)

    return w_avg
