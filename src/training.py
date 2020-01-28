from sklearn.model_selection import GroupKFold
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from src.model import PointCNN
from src import utils
from torch.utils.tensorboard import SummaryWriter
import time


def train_KFolds(train, test, target, molecules, coupling_types, batch_size=1024, n_folds=5, seed=42, debug=False):
    oof = np.zeros(len(train))
    preds = np.zeros(len(test))
    gkf = GroupKFold(n_splits=n_folds)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.L1Loss()
    utils.set_seed(seed)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(train, target, groups=molecules), 1):
        if debug:
            if fold != 1:
                continue
        xtrain, ytrain = torch.from_numpy(train[train_idx]), torch.from_numpy(target[train_idx])
        xval, yval = torch.from_numpy(train[val_idx]), torch.from_numpy(target[val_idx])
        eval_types = coupling_types[val_idx]

        train_set = TensorDataset(xtrain, ytrain)
        val_set = TensorDataset(xval, yval)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)

        model = PointCNN().to(device).float()
        optimizer = optim.Adam(model.parameters(), lr=.0003)

        summary_path = 'runs/' + time.strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(summary_path)

        for epoch in range(1, 70):
            trn_loss = 0.0
            val_loss = 0.0

            for i, (features, targets) in enumerate(train_loader):
                writer.add_graph(model, features.float())
                features, targets = features.float().to(device), targets.to(device)

                model.train()
                outputs = model(features)
                loss = criterion(outputs, targets)
                trn_loss += loss.item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                print('[%d] loss: %.5f' % (epoch, trn_loss / len(train_loader)))
                writer.add_scalar('training loss', trn_loss / len(train_loader), epoch)

                for i, (features, targets) in enumerate(val_loader):
                    features, targets = features.float().to(device), targets.to(device)

                    model.eval()
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    oof[val_idx] = outputs.cpu().numpy()
                    if i == 0:
                        eval_set = outputs.cpu().numpy()
                    else:
                        eval_set = np.vstack(eval_set, outputs.cpu.numpy())

                print('[%d] validation loss: %.5f' % (epoch, val_loss / len(val_loader)))
                writer.add_scalar('validation loss', val_loss / len(val_loader), epoch)

                val_score = utils.mean_log_mae(yval, eval_set, eval_types)
                print('[%d] validation score: %.5f' % (epoch, val_score))
                writer.add_scalar('validation score', val_score, epoch)

        writer.close()
