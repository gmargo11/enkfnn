import numpy as np
import matplotlib.pyplot as plt

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import multiprocessing
from joblib import Parallel, delayed
#import dill as pickle

import cProfile
import pstats

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, 1)
        self.conv2 = nn.Conv2d(16, 32, 5, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(512, 30)
        self.fc2 = nn.Linear(30, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=(2, 2))
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=(2, 2))
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    #model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def learn_mnist_backprop():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        scheduler.step()

    wvec = extract_weight_vector(model)
    print(len(wvec))

    perturb_noise = 0.03
    wvec = wvec + np.random.randn(wvec.shape[0]) * perturb_noise

    model = set_weights_from_vector(model, wvec)
    test(args, model, device, test_loader)

def learn_mnist_enkf_plus_backprop():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    for param in model.parameters():
        param.requires_grad = False

    ### enkf learning here
    wvec = extract_weight_vector(model)
    print(len(wvec))
    #wvec = np.random.randn(len(wvec))
    model = set_weights_from_vector(model, wvec)

    #print(predict_nn(wvec, train_loader, model))

    #meas_model = lambda wvec, xs: predict_nn(wvec, train_loader, model)
    timesteps=3001
    num_particles=200
    As = fixed_interval_smooth(wvec, train_loader, test_loader, device, args, dx=0.1, model=model, timesteps=timesteps, window=timesteps, ensemble_size=num_particles)
    print(As.keys())
    A_final = As[timesteps-1]
    wvec_final = np.mean(A_final, 1)
    model = set_weights_from_vector(model, wvec_final)

    test(args, model, device, test_loader)
    #print('Test loss (enkf):', score[0])
    #print('Test accuracy (enkf):', score[1])



    

def predict_nn(wvec, train_loader, model):
    with torch.no_grad():
        model = set_weights_from_vector(model, wvec)
        return model()

def set_weights_from_vector(model, wvec):
    state = model.state_dict()
    layer_ends = np.cumsum([w.cpu().numpy().flatten().shape[0] for w in model.state_dict().values()])
    #print(layer_ends)
    keys = list(model.state_dict().keys())
    for i in range(len(keys)):
        if i == 0:
            state[keys[i]] = torch.from_numpy(np.reshape(wvec[:layer_ends[i]], state[keys[i]].shape))
        if i >= 1:
            state[keys[i]] = torch.from_numpy(np.reshape(wvec[layer_ends[i-1]:layer_ends[i]], state[keys[i]].shape))
    #with torch.no_grad():
    model.load_state_dict(state)
    return model

def extract_weight_vector(model):
    with torch.no_grad():
        weights = list(model.parameters())
        #for w in weights:
            #print(w.cpu().numpy().shape)

        weights = np.concatenate([w.cpu().numpy().flatten() for w in weights])
    return weights

def fixed_interval_smooth(w0, train_loader, test_loader, device, args, dx, model, timesteps, window, ensemble_size):
    perturb_noise = 0.3
    dt = 0.1
    perturbation = np.random.randn(len(w0), ensemble_size) * perturb_noise
    A = np.tile(w0, (ensemble_size, 1)).T + perturbation
    Z = np.eye(ensemble_size)

    M = timesteps
    W = window
    To = range(0, M)
    Ts = range(0, M)
    X5s = {}
    As = {}

    eval_period = 10
    train_performance = np.zeros(M // eval_period)
    test_performance = np.zeros(M // eval_period)


    batch_size = 1
    i = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        print(data.shape)
        print(batch_idx)
        if batch_idx * batch_size > M:
            return As
        data, target = data.to(device), target.to(device)
        #optimizer.zero_grad()
        #output = model(data)

        for (x, y) in zip(data, target):
            i += 1
            print("==== Iteration", i, "====")

            if i % eval_period == 0 and i > 0:
                print(np.max(A))
                model = set_weights_from_vector(model, np.mean(A, 1))
                test(args, model, device, test_loader)
                model = set_weights_from_vector(model, np.mean(A, 1))
                print(np.max(A))

            X5 = np.array([])
            if i in To:
                #print(y.item())
                print(np.array([[y.item()]]))
                X5 = compute_X5_new(A, x.reshape(1, 1, 28, 28), np.array([[y.item()]]), model)
                #A = np.matmul(A, X5)
                A += X5
                #Lyerr = compute_Lyerr(A, xs[i:i+1, :, :, :], ys[i:i+1, :], meas_model)
                #A = A + Lyerr

            X5s[i] = X5

            if i in Ts:
                As[i] = A

            #if i <= W:
            #    Z = MultR(Z, X5)
            #else:
            #    Xw = X5s[i-W]
            #    Z = Mult(Xw, Z, X5)
            #    if i-W in Ts:
            #        Aw = As[i-W]
            #        Aw = np.matmul(Aw,  Z)
            #        As[i-W] = Aw
            print("Max weight:", np.max(A), "; Min weight: ", np.min(abs(A)))

    plt.figure()
    plt.plot(np.array(range(len(train_performance))) * eval_period, train_performance)
    plt.plot(np.array(range(len(test_performance))) * eval_period, test_performance)

    return As


def fixed_interval_smooth_batch(w0, xs, ys, dx, meas_model, timesteps, window, ensemble_size, batch_size, parallel=True):
    perturb_noise = 0.03
    dt = 0.1
    perturbation = np.random.randn(len(w0), ensemble_size) * perturb_noise
    A = np.tile(w0, (ensemble_size, 1)).T + perturbation
    Z = np.eye(ensemble_size)

    M = timesteps
    W = window
    To = range(1, M)
    Ts = range(1, M)
    X5s = {}
    As = {}

    eval_period = 10
    train_performance = np.zeros(M // eval_period+1)
    test_performance = np.zeros(M // eval_period+1)


    for i in range(M):
        print("==== Iteration", i, "====")
        if i % eval_period == 0 and i > 0:
            print("Evaluating...")
            print("Train Performance: ", end='')
            train_performance[i // eval_period] = evaluate_performance(A, xs[:i*batch_size, :, :, :], ys[:i*batch_size, :], meas_model);
            print("Test Performance: ", end='')
            test_performance[i // eval_period] = evaluate_performance(A, xs[M:, :, :, :], ys[M:, :], meas_model);

        X5 = np.array([])
        print("Training Sample: ", end='')


        if parallel and i in To:
            inputs = np.array(range(batch_size)) + i * batch_size
            #def process(inp):
            #    compute_X5(A, xs[inp:inp+1, :, :, :], ys[inp:inp+1, :], meas_model)

            num_cores = multiprocessing.cpu_count()
            results = Parallel(n_jobs=num_cores)(delayed(compute_X5)(A, xs[i:i+1, :, :, :], ys[i:i+1, :], meas_model) for i in inputs)

            print(len(results))

            X5 = np.mean(results, axis=0)
            print(X5.shape)
        elif i in To:
            for j in range(batch_size): # compute batch X5
                print(j, " ", end='')
                idx = i * batch_size + j
                if X5.shape[0] == 0:
                    X5 = compute_X5(A, xs[idx:idx+1, :, :, :], ys[idx:idx+1, :], meas_model)
                else:
                    X5 = X5 + compute_X5(A, xs[idx:idx+1, :, :, :], ys[idx:idx+1, :], meas_model)
            print()
            X5 = X5 / batch_size

        if i in To:
            A = np.matmul(A, X5)
        X5s[i] = X5

        if i in Ts:
            As[i] = A

        if i <= W:
            Z = MultR(Z, X5)
        else:
            Xw = X5s[i-W]
            Z = Mult(Xw, Z, X5)
            if i-W in Ts:
                Aw = As[i-W]
                Aw = np.matmul(Aw,  Z)
                As[i-W] = Aw
        print("Max weight:", np.max(A), "; Min weight: ", np.min(abs(A)))

    #plt.figure()
    #plt.plot(np.array(range(len(train_performance))) * eval_period, train_performance)
    #plt.plot(np.array(range(len(test_performance))) * eval_period, test_performance)

    return As

def compute_X5_old(A, x, y, meas_model):
    n = y.shape[0] # batch size
    l = A.shape[0] # number of weights
    q = y.shape[1] # output dimensionality
    #print(n, l, q)
    H = np.concatenate((np.eye(n*q), np.matrix(np.zeros((l, n*q))))).T

    ytilde = np.zeros((y.shape[1], A.shape[1]))
    for i in range(A.shape[1]):
        ytilde[:, i] = meas_model(A[:, i], x)
    error_y = ytilde - np.tile(np.mean(ytilde, 1), (ytilde.shape[1], 1)).T
    Pe = np.matmul(error_y, error_y.T)
    R = np.mean(np.abs(error_y))/2
    #print(R)
    #R = 0.1
    gamma = np.random.randn(ytilde.shape[0], 1) * R
    Re = np.matmul(gamma, gamma.T)

    #print("y_error shape: ", error_y.shape, "; Pe shape: ", Pe.shape)

    PRinv = np.linalg.inv(Pe+Re)
    print(np.max(PRinv))
    if np.max(np.abs(PRinv)) > 1000000000:
        print("Singularity Error")
        X5 = np.eye(A.shape[1])
    else:
        X4 = np.matmul(error_y.T, PRinv)
        #print(np.max(X4))
        X5 = np.eye(A.shape[1]) + np.matmul(X4, (np.tile(y, (ytilde.shape[1], 1)).T - ytilde))

    return X5

def compute_X5(A, x, y, model):
    #n = y.shape[0] # batch size
    #print(y)
    l = A.shape[0] # number of weights
    q = y.shape[1] # output dimensionality
    #print(n, l, q)
   # H = np.concatenate((np.eye(n*q), np.matrix(np.zeros((l, n*q))))).T

    ytilde = np.zeros((10, A.shape[1]))
    for i in range(A.shape[1]):
        #print(model(x))
        model = set_weights_from_vector(model, A[:, i])
        with torch.no_grad():
            out = model(x).cpu().numpy()
            #print(out)
            pred_idx = np.where(out[0]>-0.1)[0]
            ytilde[pred_idx, i] = 1
    #print(ytilde)
    error_y = ytilde - np.tile(np.mean(ytilde, 1), (ytilde.shape[1], 1)).T
    #print(error_y)
    #S = np.concatenate((ytilde, A))
    #error_S = S - np.tile(np.mean(S, 1), (S.shape[1], 1)).T
    #Pe = np.matmul(A.T, A)
    Pe = np.matmul(error_y, error_y.T)
    #print(error_y)
    #R = np.mean(np.abs(error_y))/2 # proportional to eigenvalue in P
    #print(R)
    w, v = np.linalg.eig(Pe)
   #R = max(w) * 0.05
    R = 5
    #print(R)
    #print(R)
    #R = 1.0
    gamma = np.random.randn(ytilde.shape[0], ytilde.shape[1]) * R
    Re = np.matmul(gamma, gamma.T)

    #print("y_error shape: ", error_y.shape, "; Pe shape: ", Pe.shape)

    #PRinv = np.linalg.inv(Pe+Re)
    try:
        PRinv = np.linalg.inv(Pe + Re)
    except np.linalg.LinAlgError:
        print("Singularity Error")
        X5 = np.eye(A.shape[1])
        return X5
    if np.max(PRinv) > 1000000000:
        print("Exploding Value Error")
        X5 = np.eye(A.shape[1])
        return X5
    
    X4 = np.matmul(error_y.T, PRinv)
    #print(np.max(X4))
    X5 = np.eye(A.shape[1]) + np.matmul(X4, (np.tile(y, (ytilde.shape[1], 1)).T + gamma - ytilde))

    return X5

def compute_X5_new(A, x, y, model):
    n = y.shape[0] # batch size
    l = A.shape[0] # number of weights
    q = y.shape[1] # output dimensionality
    #print(n, l, q)
    #H = np.concatenate((np.eye(n*q), np.matrix(np.zeros((l, n*q))))).T

    errmag = 0.003

    R = np.eye(y.shape[1]) * errmag

    ytilde = np.zeros((10, A.shape[1]))
    for i in range(A.shape[1]):
        #print(model(x))
        model = set_weights_from_vector(model, A[:, i])
        with torch.no_grad():
            output = model(x)
            #print(out)
            #pred = output.argmax(dim=1)
            #print(pred_idx)
            ytilde[:, i] = np.exp(output.cpu().numpy())

    ytilde_mean = np.mean(ytilde, 1)
    print(ytilde.shape)
    P_yy = np.zeros((ytilde.shape[0], ytilde.shape[0]))
    for i in range(A.shape[1]):
        yerr = ytilde[:, i] - ytilde_mean
        P_yy += np.outer(yerr, yerr)

    P_yy = P_yy / A.shape[1] + R

    A_mean = np.mean(A, 1)
    P_xy = np.zeros((A.shape[0], ytilde.shape[0]))
    for i in range(A.shape[1]):
        Aerr = A[:, i] - A_mean
        yerr = ytilde[:, i] - ytilde_mean
        #print(Aerr.shape, yerr.shape)
        P_xy += np.outer(Aerr, yerr)

    P_xy = P_xy / A.shape[1]

    #print("y_error shape: ", error_y.shape, "; Pe shape: ", Pe.shape)

    #PRinv = np.linalg.inv(Pe+Re)
    try:
        PRinv = np.linalg.inv(P_yy)
    except np.linalg.LinAlgError:
        print("Singularity Error")
        print(P_yy)
        #print(np.linalg.inv(P_yy))
        X5 = np.zeros_like(A)
        return X5
    if np.max(PRinv) > 1000000000:
        print("Exploding Value Error")
        X5 = np.zeros_like(A)
        return X5
    
    X4 = np.matmul(P_xy, PRinv)

    gamma = np.random.randn(ytilde.shape[0], ytilde.shape[1]) * errmag
    update = np.matmul(X4, (np.tile(y, (ytilde.shape[1], 1)).T + gamma - ytilde))

    return update

def compute_X5_old(A, x, y, meas_model):
    ytilde = np.zeros((y.shape[1], A.shape[1]))
    for i in range(A.shape[1]):
        ytilde[:, i] = meas_model(A[:, i], x)
    print(ytilde)
    print(y)
    print(np.mean(ytilde, 1))
    HAp = ytilde - np.tile(np.mean(ytilde, 1), (ytilde.shape[1], 1)).T
    Pe = np.matmul(HAp, HAp.T)# / (A.shape[1] - 1)
    R = np.mean(np.abs(HAp))
    gamma = np.random.randn(ytilde.shape[0], 1) * R
    Re = np.matmul(gamma, gamma.T)# / (A.shape[1] - 1)

    w, v = np.linalg.eig(Pe + Re)
    lambda_inv = np.diag(w) # np.linalg.inv(np.diag(w))
    U, S, V = np.linalg.svd(Pe+Re)

    X1 = np.matmul(lambda_inv, U.T)
    X2 = np.matmul(X1, (np.tile(y, (ytilde.shape[1], 1)).T - ytilde))
    X3 = np.matmul(U, X2)
    X4 = np.matmul(HAp.T, X3)
    X5 = np.eye(X4.shape[0]) + X4

    return X5

def MultR(A, B):
    if B.shape[0] > 0:
        C = A * B
    else:
        C = A
    return C

def MultL(A, B):
    if A.shape[0] > 0:
        C = np.linalg.pinv(A) * B
    else:
        C = B
    return C

def Mult(A, B, C):
    D = MultL(A, MultR(B, C))
    return D

def evaluate_performance(A, x, y, meas_model):
    params_estimate = np.mean(A, 1)
    y_predict = meas_model(params_estimate, x)
    q1 = np.argmax(y_predict, 1)
    q2 = np.argmax(y, 1)
    successes = ((q1-q2) == 0)
    success_count = np.sum(successes)
    performance = success_count / len(successes)
    #print(np.matmul(y, successes.T)/np.sum(y, 1))
    print(performance)
    return performance


if __name__ == "__main__":
    #learn_mnist_backprop()
    pr = cProfile.Profile()
    pr.enable()
    learn_mnist_enkf_plus_backprop()
    #learn_mnist_backprop()
    pr.disable()
    p = pstats.Stats(pr)
    p.sort_stats("cumulative").print_stats(50)