
"""
Implements ADDA:
Adversarial Discriminative Domain Adaptation, Tzeng et al. (2017)
"""
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from train_source import preprocess_train, preprocess_test, preprocess_train_single
import scipy
import config
from models import Net
from utils import loop_iterable, set_requires_grad, GrayscaleToRgb
import json


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EXTRACTED_FEATURE_DIM = 128
arg_parser = argparse.ArgumentParser(description='Domain adaptation using ADDA')
arg_parser.add_argument('--MODEL_FILE', type=str, default='trained_models/source.pt', help='A model in trained_models')
arg_parser.add_argument('--batch-size', type=int, default=256)
arg_parser.add_argument('--iterations', type=int, default=20)
arg_parser.add_argument('--epochs', type=int, default=20)
arg_parser.add_argument('--k-disc', type=int, default=1)
arg_parser.add_argument('--k-clf', type=int, default=3)
arg_parser.add_argument('--person', type=int, default=1)
arg_parser.add_argument('--seed', type=int, default=114514)
args = arg_parser.parse_args()
X_test = scipy.io.loadmat('../test/%d.mat'%(10 + args.person))['de_feature']
X_test = (X_test - np.min(X_test, axis=0)) / (np.max(X_test, axis=0) - np.min(X_test, axis=0))
X_test = torch.from_numpy(X_test).float().to(device)
y_label = scipy.io.loadmat('../test/%d.mat'%(10 + args.person))['label']
y_label = torch.from_numpy(y_label).long().squeeze().to(device)
torch.manual_seed(args.seed)


def test(args, model):
    X_target, y_target = preprocess_test(args.person)
    target_dataset = torch.utils.data.TensorDataset(X_target, y_target)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size,
                               num_workers=1, pin_memory=True)
    model.eval()

    total_accuracy = 0
    with torch.no_grad():
        for x, y_true in tqdm(target_loader, leave=False):
            x, y_true = x.to(device), y_true.to(device)
            y_pred = model(x)
            total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()
    
    mean_accuracy = total_accuracy / len(target_loader)
    #print(f'Accuracy on target data: {mean_accuracy:.4f}')
    return mean_accuracy


def test_all(clfs):
    preds = []
    for clf in clfs:
        pred = clf(X_test)
        pred = pred.max(1)[1]
        preds.append(pred)
    voted_pred = []
    for j in range(len(y_label)):
        label_counts = [0]*4
        for i in range(len(preds)):
            label_counts[preds[i][j]] += 1
        max_label = label_counts.index(max(label_counts))
        voted_pred.append(max_label)
    voted_pred = torch.tensor(voted_pred, dtype=torch.long).to(device)
    acc = (voted_pred == y_label).sum().item() / y_label.shape[0]
    return acc


def main(args):
    final_accs = []
    source_models = [Net().to(device) for _ in range(10)]
    for idx in range(len(source_models)):
        source_models[idx].load_state_dict(torch.load(args.MODEL_FILE))
        source_models[idx].eval()
        set_requires_grad(source_models[idx], requires_grad=False)
    
    clfs = [source_model for source_model in source_models]
    source_models = [source_model.feature_extractor for source_model in source_models]

    target_models = [Net().to(device) for _ in range(10)]
    for idx in range(len(target_models)):
        target_models[idx].load_state_dict(torch.load(args.MODEL_FILE))
        target_models[idx] = target_models[idx].feature_extractor

    discriminators = [nn.Sequential(
        nn.Linear(EXTRACTED_FEATURE_DIM, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Linear(64, 1),
        nn.Sigmoid()
    ).to(device) for _ in range(10)]

    batch_size = args.batch_size
    discriminator_optims = [torch.optim.Adam(discriminators[idx].parameters(), lr=1e-5) for idx in range(10)]
    target_optims = [torch.optim.Adam(target_models[idx].parameters(), lr=1e-5) for idx in range(10)]
    criterion = nn.BCEWithLogitsLoss()

    source_loaders = []
    target_loaders = []
    for idx in range(10):
        X_source, y_source = preprocess_train_single(idx)
        source_dataset = torch.utils.data.TensorDataset(X_source, y_source)

        source_loader = DataLoader(source_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=1, pin_memory=True)
        source_loaders.append(source_loader)
    
        X_target, y_target = preprocess_test(args.person)
        target_dataset = torch.utils.data.TensorDataset(X_target, y_target)
        target_loader = DataLoader(target_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=1, pin_memory=True)
        target_loaders.append(target_loader)


    best_voting_acc = test_all(clfs)
    best_tar_accs = [0.0]*10

    for epoch in range(1, args.epochs+1):
        source_loaders = [DataLoader(source_loaders[idx].dataset, batch_size=batch_size, shuffle=True) for idx in range(10)]
        target_loaders = [DataLoader(target_loaders[idx].dataset, batch_size=batch_size, shuffle=True) for idx in range(10)]
        for idx in range(10):
            source_loader = source_loaders[idx]
            target_loader = target_loaders[idx]
            batch_iterator = zip(loop_iterable(source_loader), loop_iterable(target_loader))
            target_model = target_models[idx]
            discriminator = discriminators[idx]
            source_model = source_models[idx]
            clf = clfs[idx]
            total_loss = 0
            adv_loss = 0
            total_accuracy = 0
            second_acc = 0
            for _ in trange(args.iterations, leave=False):
                # Train discriminator
                set_requires_grad(target_model, requires_grad=False)
                set_requires_grad(discriminator, requires_grad=True)
                discriminator.train()
                for _ in range(args.k_disc):
                    (source_x, _), (target_x, _) = next(batch_iterator)
                    source_x, target_x = source_x.to(device), target_x.to(device)

                    source_features = source_model(source_x).view(source_x.shape[0], -1)
                    target_features = target_model(target_x).view(target_x.shape[0], -1)

                    discriminator_x = torch.cat([source_features, target_features])
                    discriminator_y = torch.cat([torch.ones(source_x.shape[0], device=device),
                                                torch.zeros(target_x.shape[0], device=device)])

                    preds = discriminator(discriminator_x).squeeze()
                    loss = criterion(preds, discriminator_y)

                    discriminator_optims[idx].zero_grad()
                    loss.backward()
                    discriminator_optims[idx].step()

                    total_loss += loss.item()
                    total_accuracy += ((preds >= 0.5).long() == discriminator_y.long()).float().mean().item()

                # Train classifier
                set_requires_grad(target_model, requires_grad=True)
                set_requires_grad(discriminator, requires_grad=False)
                target_model.train()
                for _ in range(args.k_clf):
                    _, (target_x, _) = next(batch_iterator)
                    target_x = target_x.to(device)
                    target_features = target_model(target_x).view(target_x.shape[0], -1)

                    # flipped labels
                    discriminator_y = torch.ones(target_x.shape[0], device=device)

                    preds = discriminator(target_features).squeeze()
                    second_acc += ((preds >= 0.5).long() == discriminator_y.long()).float().mean().item()
                    
                    loss = criterion(preds, discriminator_y)
                    adv_loss += loss.item()

                    target_optims[idx].zero_grad()
                    loss.backward()
                    target_optims[idx].step()

            mean_loss = total_loss / (args.iterations*args.k_disc)
            mean_adv_loss = adv_loss / (args.iterations * args.k_clf)
            dis_accuracy = total_accuracy / (args.iterations*args.k_disc)
            sec_acc = second_acc / (args.iterations * args.k_clf)
            clf.feature_extractor = target_model
            tar_accuarcy = test(args, clf)
            if tar_accuarcy > best_tar_accs[idx]:
                best_tar_accs[idx] = tar_accuarcy
                torch.save(clf.state_dict(), 'trained_models/adda'+str(idx)+'.pt')
            
            tqdm.write(f'EPOCH {epoch:03d}: discriminator_loss={mean_loss:.4f}, adv_loss = {mean_adv_loss:.4f}, '
                    f'discriminator_accuracy={dis_accuracy:.4f}, tar_accuary = {tar_accuarcy:.4f}, best_accuracy = {best_tar_accs[idx]:.4f}, sec_acc = {sec_acc:.4f}')

            # Create the full target model and save it
            clf.feature_extractor = target_model
            #torch.save(clf.state_dict(), 'trained_models/adda.pt')
        acc = test_all(clfs)
        final_accs.append(acc)
        if acc > best_voting_acc:
            best_voting_acc = acc
        print("In epoch %d, voting_acc: %.4f, best_voting_acc: %.4f" %(epoch, acc, best_voting_acc))
    jd = {"test_acc": final_accs}
    with open(str(args.seed)+'/acc'+str(args.person)+'.json', 'w') as f:
        json.dump(jd, f)


if __name__ == '__main__':
    main(args)
