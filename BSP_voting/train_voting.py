import torch
import torch.optim as optim
import torch.nn as nn
import model
import adversarial1 as ad
import numpy as np
import argparse
import os
import torch.nn.functional as F
import scipy.io
from tqdm import tqdm
import json


torch.set_num_threads(1)
torch.manual_seed(1)

parser = argparse.ArgumentParser(description='PyTorch BSP Example')
parser.add_argument('--gpu_id', type=str, nargs='?', default='1', help="device id to run")
parser.add_argument('--num_iter', type=int, default=30000, help='max iter_num')
parser.add_argument('--person', type=int, default=1, help='which person in the target domain to train')
parser.add_argument('--person2', type=int, default=0, help='which person in the source domain to train(when using voting)')
parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
parser.add_argument('--bsp', type=float, default=0.0001)
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLF_RATIO = 1
TRANS_RATIO = 1


def dataset_load(batch_size = 64, person=args.person):
    X_source = np.array([])
    y_source = np.array([])

    for i in range(10):
        data = scipy.io.loadmat('../train/%d.mat'%(i+1))['de_feature']
        label = scipy.io.loadmat('../train/%d.mat'%(i+1))['label']
        
        if i == 0:
            X_source = data
            y_source = label
        else:
            X_source = np.vstack((X_source, data))
            y_source = np.vstack((y_source, label))

    X_source = (X_source - np.min(X_source, axis=0)) / (np.max(X_source, axis=0) - np.min(X_source, axis=0))
    X_source = torch.from_numpy(X_source).float()
    y_source = torch.from_numpy(y_source).long().squeeze()
    source_dataset = torch.utils.data.TensorDataset(X_source, y_source)

    X_target = scipy.io.loadmat('../test/%d.mat'%(10 + person))['de_feature']
    y_target = scipy.io.loadmat('../test/%d.mat'%(10 + person))['label']
    X_target = (X_target - np.min(X_target, axis=0)) / (np.max(X_target, axis=0) - np.min(X_target, axis=0))
    X_target = torch.from_numpy(X_target).float()
    y_target = torch.from_numpy(y_target).long().squeeze()
    target_dataset = torch.utils.data.TensorDataset(X_target, y_target)

    return source_dataset, target_dataset


def dataset_load_single(batch_size = 64, person=args.person, person2=0):
    X_source = np.array([])
    y_source = np.array([])

    data = scipy.io.loadmat('../train/%d.mat'%(person2+1))['de_feature']
    label = scipy.io.loadmat('../train/%d.mat'%(person2+1))['label']    
    X_source = data
    y_source = label

    X_source = (X_source - np.min(X_source, axis=0)) / (np.max(X_source, axis=0) - np.min(X_source, axis=0))
    X_source = torch.from_numpy(X_source).float()
    y_source = torch.from_numpy(y_source).long().squeeze()
    source_dataset = torch.utils.data.TensorDataset(X_source, y_source)

    X_target = scipy.io.loadmat('../test/%d.mat'%(10 + person))['de_feature']
    y_target = scipy.io.loadmat('../test/%d.mat'%(10 + person))['label']
    X_target = (X_target - np.min(X_target, axis=0)) / (np.max(X_target, axis=0) - np.min(X_target, axis=0))
    X_target = torch.from_numpy(X_target).float()
    y_target = torch.from_numpy(y_target).long().squeeze()
    target_dataset = torch.utils.data.TensorDataset(X_target, y_target)

    return source_dataset, target_dataset


def preprocess_test(person = 3):
    X_test = scipy.io.loadmat('../test/%d.mat'%(10 + person))['de_feature']
    y_test = scipy.io.loadmat('../test/%d.mat'%(10 + person))['label']
    X_test = (X_test - np.min(X_test, axis=0)) / (np.max(X_test, axis=0) - np.min(X_test, axis=0))
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long().squeeze()

    return X_test, y_test


def test(model):
    X_target, y_target = preprocess_test(args.person)
    target_dataset = torch.utils.data.TensorDataset(X_target, y_target)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False,
                               num_workers=1, pin_memory=True)
    model.eval()

    total_accuracy = 0
    preds = []
    with torch.no_grad():
        for x, y_true in tqdm(target_loader, leave=False):
            x, y_true = x.to(device), y_true.to(device)
            y_pred = model(x)
            preds += y_pred.max(1)[1].tolist()
            total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()
    # print(preds)
    mean_accuracy = total_accuracy / len(target_loader)
    model.train()
    #print(f'Accuracy on target data: {mean_accuracy:.4f}')
    return mean_accuracy


def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001, weight_decay=0.0005):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        param_group['weight_decay'] = weight_decay * 2
        i += 1
    return optimizer


class BSP_CDAN(nn.Module):
    def __init__(self, num_features):
        super(BSP_CDAN, self).__init__()
        self.model_fc = model.FeatureExtractor(num_features)
        self.bottleneck_layer1 = nn.Linear(num_features, 256)
        self.bottleneck_layer1.apply(init_weights)
        self.bottleneck_layer = nn.Sequential(self.bottleneck_layer1, nn.ReLU(), nn.Dropout(0.5))
        self.classifier_layer = nn.Linear(256, 4)
        self.classifier_layer.apply(init_weights)
        self.predict_layer = nn.Sequential(self.model_fc, self.bottleneck_layer, self.classifier_layer)

    def forward(self, x):
        feature = self.model_fc(x)
        out = self.bottleneck_layer(feature)
        outC = self.classifier_layer(out)
        return (out, outC)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)

def BSP(feature):
    feature_s = feature.narrow(0, 0, int(feature.size(0) / 2))
    feature_t = feature.narrow(0, int(feature.size(0) / 2), int(feature.size(0) / 2))
    _, s_s, _ = torch.svd(feature_s)
    _, s_t, _ = torch.svd(feature_t)
    sigma = torch.pow(s_s[0], 2) + torch.pow(s_t[0], 2)
    return sigma

def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target)

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


if __name__ == "__main__":
    # source_dataset, target_dataset = dataset_load()
    source_dataset, target_dataset = dataset_load_single(person2=args.person2)
    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=args.batch_size, shuffle=False)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False)

    max_iter = args.num_iter
    accs = []
    num_features = 256
    net = BSP_CDAN(num_features)
    net = net.to(device)
    ad_net = AdversarialNetwork(num_features*4, 1024)
    ad_net = ad_net.to(device)
    net.train(True)
    ad_net.train(True)
    criterion = {"classifier": nn.CrossEntropyLoss(), "adversarial": nn.BCELoss()}
    optimizer_dict = [{"params": filter(lambda p: p.requires_grad, net.model_fc.parameters()), "lr": 0.1},
                        {"params": filter(lambda p: p.requires_grad, net.bottleneck_layer.parameters()), "lr": 1},
                        {"params": filter(lambda p: p.requires_grad, net.classifier_layer.parameters()), "lr": 1},
                        {"params": filter(lambda p: p.requires_grad, ad_net.parameters()), "lr": 1}]
    optimizer = optim.SGD(optimizer_dict, lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
    # optimizer = optim.Adam(net.parameters())
    train_cross_loss = train_transfer_loss = train_total_loss = train_sigma = 0.0
    len_source = len(source_loader) - 1
    len_target = len(target_loader) - 1
    param_lr = []
    iter_source = iter(source_loader)
    iter_target = iter(target_loader)
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    test_interval = 100
    num_iter = max_iter
    for iter_num in range(1, num_iter + 1):
        net.train(True)
        optimizer = inv_lr_scheduler(param_lr, optimizer, iter_num, init_lr=0.003, gamma=0.0001, power=0.75,
                                weight_decay=0.0005)
        optimizer.zero_grad()
        if iter_num % len_source == 0:
            iter_source = iter(source_loader)
        if iter_num % len_target == 0:
            iter_target = iter(target_loader)
        data_source = iter_source.next()
        data_target = iter_target.next()
        inputs_source, labels_source = data_source
        inputs_target, labels_target = data_target
        inputs = torch.cat((inputs_source, inputs_target), dim=0)
        inputs = inputs.to(device)
        labels = labels_source.to(device)
        feature, outC = net(inputs)
        feature_s = feature.narrow(0, 0, int(feature.size(0) / 2))
        feature_t = feature.narrow(0, int(feature.size(0) / 2), int(feature.size(0) / 2))
        _, s_s, _ = torch.svd(feature_s)
        _, s_t, _ = torch.svd(feature_t)
        sigma = torch.pow(s_s[0], 2) + torch.pow(s_t[0], 2)
        sigma_loss = args.bsp * sigma
        classifier_loss = criterion["classifier"](outC.narrow(0, 0, args.batch_size), labels)
        # total_loss = classifier_loss
        softmax_out = nn.Softmax(dim=1)(outC)
        entropy = Entropy(softmax_out)
        coeff = calc_coeff(iter_num)
        transfer_loss = CDAN([feature, softmax_out], ad_net, entropy, coeff, random_layer=None)
        total_loss = CLF_RATIO*classifier_loss + TRANS_RATIO*transfer_loss + sigma_loss
        total_loss.backward()
        optimizer.step()
        train_cross_loss += classifier_loss.item()
        train_transfer_loss += transfer_loss.item()
        train_total_loss += total_loss.item()
        train_sigma += sigma_loss.item()
        if iter_num % test_interval == 0:
            print(
            "Iter {:05d}, Average Cross Entropy Loss: {:.4f}; Average Transfer Loss: {:.4f}; Average Sigma Loss: {:.9f}; Average Training Loss: {:.4f}".format(
                iter_num, train_cross_loss / float(test_interval), train_transfer_loss / float(test_interval),
                            train_sigma / float(test_interval),
                            train_total_loss / float(test_interval)))
            train_cross_loss = train_transfer_loss = train_total_loss = train_sigma = 0.0
        if (iter_num % test_interval) == 0:
            net.eval()
            test_acc = test(net.predict_layer)
            accs.append(test_acc)
            print('test_acc:%.4f'%(test_acc))
    jd = {"test_acc": accs}
    with open('acc'+str(args.person)+'.json', 'w') as f:
        json.dump(jd, f)
