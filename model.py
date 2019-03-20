import os

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import requests
import scipy.sparse as sps
import sklearn.neighbors as nbrs
import torch
import torch.optim as optim
import torch.nn as nn
import torch.sparse as tsps
import torch_sparse
import torch_scatter as tsct
import torch_geometric.nn as gnn
import torch_geometric

from surprise import Dataset, KNNBasic, Reader
from surprise.model_selection import KFold, train_test_split

torch.set_default_tensor_type('torch.cuda.FloatTensor')


class RGCNNFactorization(nn.Module):

    def __init__(self, input_shape, factorization_rank=10, n_channels=32,
                 basis_order=5, diffusion_time=10, hidden_cells=32,
                 lstm_layers=1, bidirectional=False):
        super().__init__()

        # GCNN parameters
        self.m = input_shape[0]
        self.n = input_shape[1]
        self.r = factorization_rank
        self.q = n_channels
        self.order = basis_order

        # LSTM parameters
        self.nH = hidden_cells
        self.T = diffusion_time
        self.n_layers = lstm_layers
        self.bidirectional = False

        # GCNNs for H, W matrices
        self.hconv = gnn.ChebConv(in_channels=self.r, out_channels=self.q,
                                  K=self.order)
        self.wconv = gnn.ChebConv(in_channels=self.r, out_channels=self.q,
                                  K=self.order)

        # RNN
        self.lstm = nn.LSTM(input_size=self.q, hidden_size=self.nH,
                            num_layers=self.n_layers, bidirectional=self.bidirectional,
                            batch_first=True)

        self.dense_H = nn.Linear(in_features=self.nH, out_features=self.r)

        self.dense_W = nn.Linear(in_features=self.nH, out_features=self.r)

        self.loss_fn = recommender_loss

    def init_hidden(self):
        h0 = torch.zeros((self.q,)).view(1, 1, -1)
        c0 = torch.zeros((self.q,)).view(1, 1, -1)
        return h0, c0

    def forward(self, H, W, HA, WA):
        hidden = self.init_hidden()
        Hout = H
        Wout = W
        for i in range(self.T):
            conv1 = self.hconv(Hout, HA)
            Htilde = torch.sigmoid(conv1)
            out, hidden = self.lstm(Htilde.unsqueeze(0), hidden)
            dout = self.dense_H(out)
            dH = torch.tanh(dout).squeeze()
            Hout = Hout + dH

            conv2 = self.wconv(Wout, WA)
            Wtilde = torch.sigmoid(conv2)
            out, hidden = self.lstm(Wtilde.unsqueeze(0), hidden)
            dout = self.dense_W(out)
            dW = torch.tanh(dout).squeeze()
            Wout = Wout + dW

        return Hout, Wout

    def train(self, H, W, HA, WA, Y,  Xorig, Test, iters, optimizer=None):
        if optimizer is None:
            optimizer = optim.Adam(self.parameters(), lr=1e-4)

        loss_history = np.zeros((iters,))
        error_history = np.zeros((iters,))

        testmask = sps.coo_matrix((np.ones_like(Test.data), (Test.row, Test.col)))
        testmask.resize(Xorig.shape)
        testmask = tensor_from_scipy_sparse(testmask).cuda()
        Test = tensor_from_scipy_sparse(Test).cuda()

        maximum = np.max(Y)
        minimum = np.min(Y)
        M = sps.coo_matrix(Y + Xorig)
        allmask = tensor_from_scipy_sparse(sps.coo_matrix((np.ones_like(M.data), (M.row, M.col)))).cuda()
        # Y = tensor_from_scipy_sparse(Y).cuda()
        M = tensor_from_scipy_sparse(M).cuda()

        Lh = sps.csgraph.laplacian(HA)
        Lw = sps.csgraph.laplacian(WA)
        Lh = tensor_from_scipy_sparse(Lh).cuda()
        Lw = tensor_from_scipy_sparse(Lw).cuda()

        H = torch.tensor(H).float().cuda()
        W = torch.tensor(W).float().cuda()
        HA = torch.tensor([HA.row, HA.col]).long().cuda()
        WA = torch.tensor([WA.row, WA.col]).long().cuda()

        for i in range(iters):

            Hout, Wout = self.forward(H, W, HA, WA)
            loss = self.loss_fn((Hout, Wout), (Lh, Lw), M, allmask, (minimum, maximum))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_history[i] = loss.item()
            Xpred = combine(Hout, Wout)
            error_history[i] = self.predict(Xpred, Test, testmask)
            print('iter %s: loss: %s, error: %s' % (i, loss_history[i], error_history[i]))

        return Hout.cpu().detach(), Wout.cpu().detach(), loss_history, error_history

    def predict(self, X, Y, Mask=None, norm=False):
        if norm:
            X = 1 + 4 * (X - torch.min(X)) / (torch.max(X) - torch.min(X))
        if Mask is None:
            Mask = sps.coo_matrix((np.ones_like(Y.data), (Y.row, Y.col)), shape=Y.shape)
            Mask = tensor_from_scipy_sparse(Mask)
            Y = tensor_from_scipy_sparse(Y)
        predictions = X * Mask.to_dense()
        predictions_error = torch.sum(torch.abs(predictions - Y)) / torch.sum(Mask.to_dense())
        return predictions_error.item()


def combine(H, W, minimum=1, maximum=5):
    Xpred = torch.mm(H, torch.transpose(W, 0, 1))
    Xpred = minimum + (maximum - 1) * (Xpred - torch.min(Xpred)) / (torch.max(Xpred - torch.min(Xpred)))
    return Xpred


# for computation of loss
def frobenius_norm(x):
    """norm for matrices"""
    x2 = x ** 2
    x2sum = torch.sum(x2)
    return torch.sqrt(x2sum)


def graph_norm(X, laplacian):
    norm = torch.mm(torch.transpose(X, 0, 1), tsps.mm(laplacian, X))
    return norm


def recommender_loss(inputs, laplacians, target, mask, extrema, gamma=1e-10):
    # loss function borrowed from objective in Srebro et. al 2004.
    H, W = inputs
    Lh, Lw = laplacians

    # set X to valid ratings
    X_normed = combine(H, W)

    # consider only original data and test data, ignore other sparse values.
    xm = mask.to_dense() * (X_normed - target)
    fnorm = frobenius_norm(xm)
    fnorm = fnorm / torch.sum(mask.to_dense())

    # compute regularization
    gH = graph_norm(H, Lh)
    gW = graph_norm(W, Lw)
    loss = fnorm + (gamma / 2) * (torch.trace(gH) + torch.trace(gW))
    return loss


def tensor_from_scipy_sparse(X):
    values = X.data
    indices = np.vstack((X.row, X.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    X = torch.sparse.FloatTensor(i, v, torch.Size(X.shape))
    return X


if __name__ == '__main__':
    item = 'u.item'
    url = 'http://files.grouplens.org/datasets/movielens/ml-100k/%s' % item
    if not os.path.exists(item):
        r = requests.get(url, stream=True)
        print('downloading file...')
        with open(item, 'w') as fd:
            for content in r.iter_content():
                fd.write(str(content, encoding='latin1'))
    item_features = pd.read_csv('u.item', sep='|', header=None, names=[
        'movie id', 'movie title', 'release date', 'video release date',
        'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
        'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama',
        'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
        'Sci-Fi', 'Thriller', 'War', 'Western'])
    item_features.drop(columns=['video release date', 'IMDb URL'], inplace=True)
    print('done!')

    features = item_features[['unknown', 'Action', 'Adventure', 'Animation',
                              'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama',
                              'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
                              'Sci-Fi', 'Thriller', 'War', 'Western']]
    # WA = nbrs.kneighbors_graph(features, n_neighbors=10)

    #kf = KFold(n_splits=5)
    #movielens = Dataset.load_builtin('ml-100k')
    #train, test = next(kf.split(movielens))
    train_df = pd.read_csv('ua.base', sep="\t", names=['uid', 'iid', 'rating', 'time'])
    test_df = pd.read_csv('ua.test', sep='\t', names=['uid', 'iid', 'rating', 'time'])

    fields = ['uid', 'iid', 'rating']
    reader = Reader(rating_scale=(1, 5))
    train_set = Dataset.load_from_df(df=train_df[fields], reader=reader)
    test = Dataset.load_from_df(df=test_df[fields], reader=reader).build_full_trainset()
    np.random.seed(0)
    train, validate = train_test_split(train_set, test_size=0.2)

    i, j, data = zip(*train.all_ratings())
    print(len(data))
    X = sps.coo_matrix((data, (i, j)))
    i, j, data = zip(*validate)
    i = [int(ia - 1) for ia in i]
    j = [int(ja - 1) for ja in j]
    Y = sps.coo_matrix((data, (i, j)))
    X.resize(Y.shape)

    i, j, data = zip(*test.all_ratings())
    Test = sps.coo_matrix((data, (i, j)))
    Test.resize(X.shape)

    HA = nbrs.kneighbors_graph(X, n_neighbors=10)
    WA = nbrs.kneighbors_graph(X.T, n_neighbors=10)

    k = 10
    U, Sigma, V = sps.linalg.svds(X, k=k)
    H = U * Sigma
    W = V.T * Sigma
    HA = sps.coo_matrix(HA)
    WA = sps.coo_matrix(WA)
    Xorig = X.copy()
    Xorig.resize(Y.shape)
    #M = sps.coo_matrix(Xtest + Y)
    torch.initial_seed()
    model = RGCNNFactorization(X.shape, factorization_rank=k)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    fig = plt.figure()
    #modelN = 8000
    from glob import glob
    # model_path = glob('models/model%s_*.pickle' % modelN)[0]
    # model.load_state_dict(torch.load(model_path))
    epochs = 10
    loss_history = []
    error_history = []
    for i in range(30):
        Hout, Wout, lhistory, ehistory = model.train(H, W, HA, WA, Y, Xorig, Test, epochs, optimizer=optimizer)
        loss_history += lhistory.tolist()
        error_history += ehistory.tolist()
        torch.save(model.state_dict(), 'models2/model%s_%.4f.pickle' % (epochs * (i + 1), ehistory[-1]))
        X = combine(Hout, Wout)
        plt.imshow(X, cmap='hot', vmin=1, vmax=5)
        plt.savefig('figs2/plot%s_%.4f.png' % (epochs * (i + 1), ehistory[-1]))
    plt.subplot(1, 2, 1)
    plt.plot([10 * i for i in range(len(loss_history))], loss_history)
    plt.ylabel('loss')
    plt.xlabel('iter')
    plt.subplot(1, 2, 2)
    plt.plot([10 * i for i in range(len(error_history))], error_history)
    plt.ylabel('MAE')
    plt.xlabel('iter')
    plt.show()
    print(loss_history)

    X = combine(Hout, Wout)
    plt.subplot(1, 2, 1)
    plt.imshow(X, cmap='hot', vmin=1, vmax=5)
    plt.subplot(1, 2, 2)

    X.numpy()
    with open('X.pickle', 'wb') as fd:
        pickle.dump(X, fd, pickle.HIGHEST_PROTOCOL)

    # combine X, Test with forward pass and dump data...
    error = model.predict(X, Test)
    print(error)

