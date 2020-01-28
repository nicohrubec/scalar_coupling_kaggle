import torch
import torch.backends.cudnn
import numpy as np
import pandas as pd
from sklearn import metrics


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def mean_log_mae(ytrue, ypred, groups):
    ytrue, ypred, groups = np.reshape(ytrue, (len(ytrue), 1)), np.reshape(ypred, (len(ypred), 1)), \
                           np.reshape(groups, (len(groups), 1))
    stack = np.hstack((groups, ypred, ytrue))
    df = pd.DataFrame(data=stack, columns=['groups', 'ypred', 'ytrue'])
    maes = []

    for t in df.groups.unique():
        ytrue_t = df[df.groups == t].ytrue.values
        ypred_t = df[df.groups == t].ypred.values
        mae = np.log(metrics.mean_absolute_error(ytrue_t, ypred_t))
        maes.append(mae)
    return np.mean(maes)
