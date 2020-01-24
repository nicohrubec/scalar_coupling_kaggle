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


def train_KFolds(train, test, target, molecules, n_folds=5, seed=42, debug=False):
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

        train_set = TensorDataset(xtrain, ytrain)
        val_set = TensorDataset(xval, yval)

        train_loader = DataLoader(train_set, batch_size=4)
        val_loader = DataLoader(val_set, batch_size=4)

        model = PointCNN().to(device).float()
        optimizer = optim.Adam(model.parameters(), lr=.0003)

        summary_path = 'runs/' + time.strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(summary_path)

        for epoch in range(1, 10):
            trn_loss = 0.0
            val_loss = 0.0

            for i, (features, targets) in enumerate(train_loader):
                writer.add_graph(model, features.float())
                if i > 0:
                    break
                features, targets = features.float().to(device), targets.to(device)

                model.train()
                outputs = model(features)
                loss = criterion(outputs, targets)
                trn_loss += loss.item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                print(trn_loss)
                print(outputs)
                print(targets)

            with torch.no_grad():
                print('[%d] loss: %.5f' % (epoch, trn_loss / len(train_loader)))
                writer.add_scalar('training loss', trn_loss, epoch)

                for i, (features, targets) in enumerate(val_loader):
                    if i > 0:
                        break
                    features, targets = features.float().to(device), targets.to(device)

                    model.eval()
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    oof[(i * 4):((i+1)*4)] = outputs.cpu().numpy()

                print('[%d] validation loss: %.5f' % (epoch, val_loss / len(val_loader)))
        writer.close()
