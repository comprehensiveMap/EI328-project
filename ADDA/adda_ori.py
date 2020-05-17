"""
Implements ADDA:
Adversarial Discriminative Domain Adaptation, Tzeng et al. (2017)
"""
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from train_source import preprocess_train, preprocess_test, preprocess_train_single

import config
from models import Net
from utils import loop_iterable, set_requires_grad, GrayscaleToRgb
torch.manual_seed(1919810)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EXTRACTED_FEATURE_DIM = 128

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


def gen_pred(args, model):
    X_target, y_target = preprocess_test(args.person)
    target_dataset = torch.utils.data.TensorDataset(X_target, y_target)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size,
                               num_workers=1, pin_memory=True)
    model.eval()
    preds = []
    with torch.no_grad():
        for x, y_true in tqdm(target_loader, leave=False):
            x, y_true = x.to(device), y_true.to(device)
            y_pred = model(x).tolist()
            preds.append(y_pred)
    return preds


def main(args):
    source_model = Net().to(device)
    source_model.load_state_dict(torch.load(args.MODEL_FILE))
    source_model.eval()
    set_requires_grad(source_model, requires_grad=False)
    
    clf = source_model
    source_model = source_model.feature_extractor

    target_model = Net().to(device)
    target_model.load_state_dict(torch.load(args.MODEL_FILE))
    target_model = target_model.feature_extractor

    discriminator = nn.Sequential(
        nn.Linear(EXTRACTED_FEATURE_DIM, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Linear(64, 1),
        nn.Sigmoid()
    ).to(device)

    #half_batch = args.batch_size // 2

    batch_size = args.batch_size

    # X_source, y_source = preprocess_train()
    X_source, y_source = preprocess_train_single(1)
    source_dataset = torch.utils.data.TensorDataset(X_source, y_source)

    source_loader = DataLoader(source_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=1, pin_memory=True)
    
    X_target, y_target = preprocess_test(args.person)
    target_dataset = torch.utils.data.TensorDataset(X_target, y_target)
    target_loader = DataLoader(target_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=1, pin_memory=True)

    discriminator_optim = torch.optim.Adam(discriminator.parameters())
    target_optim = torch.optim.Adam(target_model.parameters(), lr=3e-6)
    criterion = nn.BCEWithLogitsLoss()

    best_tar_acc = test(args, clf)

    for epoch in range(1, args.epochs+1):
        source_loader = DataLoader(source_loader.dataset, batch_size=batch_size, shuffle=True)
        target_loader = DataLoader(target_loader.dataset, batch_size=batch_size, shuffle=True)
        batch_iterator = zip(loop_iterable(source_loader), loop_iterable(target_loader))

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

                discriminator_optim.zero_grad()
                loss.backward()
                discriminator_optim.step()

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

                target_optim.zero_grad()
                loss.backward()
                target_optim.step()

        mean_loss = total_loss / (args.iterations*args.k_disc)
        mean_adv_loss = adv_loss / (args.iterations * args.k_clf)
        dis_accuracy = total_accuracy / (args.iterations*args.k_disc)
        sec_acc = second_acc / (args.iterations * args.k_clf)
        clf.feature_extractor = target_model
        tar_accuarcy = test(args, clf)
        if tar_accuarcy > best_tar_acc:
            best_tar_acc = tar_accuarcy
            torch.save(clf.state_dict(), 'trained_models/adda.pt')
        
        tqdm.write(f'EPOCH {epoch:03d}: discriminator_loss={mean_loss:.4f}, adv_loss = {mean_adv_loss:.4f}, '
                   f'discriminator_accuracy={dis_accuracy:.4f}, tar_accuary = {tar_accuarcy:.4f}, best_accuracy = {best_tar_acc:.4f}, sec_acc = {sec_acc:.4f}')

        # Create the full target model and save it
        clf.feature_extractor = target_model
        #torch.save(clf.state_dict(), 'trained_models/adda.pt')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Domain adaptation using ADDA')
    arg_parser.add_argument('MODEL_FILE', help='A model in trained_models')
    arg_parser.add_argument('--batch-size', type=int, default=128)
    arg_parser.add_argument('--iterations', type=int, default=50)
    arg_parser.add_argument('--epochs', type=int, default=30)
    arg_parser.add_argument('--k-disc', type=int, default=1)
    arg_parser.add_argument('--k-clf', type=int, default=3)
    arg_parser.add_argument('--person', type=int, default=3)
    args = arg_parser.parse_args()
    main(args)
