import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
import scipy.io

import config
from models import Net


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def preprocess_train():
    X_train = np.array([])
    y_train = np.array([])

    for i in range(10):
        data = scipy.io.loadmat('../train/%d.mat'%(i+1))['de_feature']
        label = scipy.io.loadmat('../train/%d.mat'%(i+1))['label']
        
        if i == 0:
            X_train = data
            y_train = label
        else:
            X_train = np.vstack((X_train, data))
            y_train = np.vstack((y_train, label))

    X_train = (X_train - np.min(X_train, axis=0)) / (np.max(X_train, axis=0) - np.min(X_train, axis=0))

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long().squeeze()

    return X_train, y_train

def preprocess_test(person = 3):
    X_test = scipy.io.loadmat('../test/%d.mat'%(10 + person))['de_feature']
    y_test = scipy.io.loadmat('../test/%d.mat'%(10 + person))['label']
    X_test = (X_test - np.min(X_test, axis=0)) / (np.max(X_test, axis=0) - np.min(X_test, axis=0))
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long().squeeze()

    return X_test, y_test

def create_dataloaders(batch_size):
    X_train, y_train = preprocess_train()
    X_test, y_test = preprocess_test()
    train_dataset=torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)


    #如果单纯用source来训练模型就使用下面的
    shuffled_indices = np.random.permutation(len(train_dataset))
    train_idx = shuffled_indices[:int(0.8*len(train_dataset))]
    val_idx = shuffled_indices[int(0.8*len(train_dataset)):]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True,
                              sampler=SubsetRandomSampler(train_idx),
                              num_workers=1, pin_memory=True)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False,
                            sampler=SubsetRandomSampler(val_idx),
                            num_workers=1, pin_memory=True)

    #如果想要用test来训练初始模型就使用下面的
    '''
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle = True,
                              num_workers=1, pin_memory=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False, shuffle = False,
                            num_workers=1, pin_memory=True)
    '''
    return train_loader, val_loader


def do_epoch(model, dataloader, criterion, optim=None):
    total_loss = 0
    total_accuracy = 0
    for x, y_true in tqdm(dataloader, leave=False):
        x, y_true = x.to(device), y_true.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y_true)

        if optim is not None:
            optim.zero_grad()
            loss.backward()
            optim.step()

        total_loss += loss.item()
        total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()
    mean_loss = total_loss / len(dataloader)
    mean_accuracy = total_accuracy / len(dataloader)

    return mean_loss, mean_accuracy


def main(args):
    train_loader, val_loader = create_dataloaders(args.batch_size)

    model = Net().to(device)
    optim = torch.optim.Adam(model.parameters())
    lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss, train_accuracy = do_epoch(model, train_loader, criterion, optim=optim)

        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy = do_epoch(model, val_loader, criterion, optim=None)

        tqdm.write(f'EPOCH {epoch:03d}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f} '
                   f'val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')

        if val_accuracy > best_accuracy:
            print('Saving model...')
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'trained_models/source.pt')

        lr_schedule.step(val_loss)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Train a classifier on source domain')
    arg_parser.add_argument('--batch-size', type=int, default=64)
    arg_parser.add_argument('--epochs', type=int, default=30)
    args = arg_parser.parse_args()
    main(args)
