import numpy as np
import pandas as pd
import pickle
import torch as pt

from torch.autograd import Variable
from torch import sigmoid
from torch.nn.functional import binary_cross_entropy


class Predictor:

    def __init__(self, mun_votes, mun_info, d, gamma=0.01, n_iter=20000):
        # Drop rows containing nans due to mergers.
        mun_votes = mun_votes.dropna(axis=0)
        mun_info = mun_info.loc[mun_votes.index]
        # Factorize and retrieve features and weights.
        U, s, V = np.linalg.svd(mun_votes.values.astype(np.float))
        self.Xs = pd.DataFrame(U[:, :d] * s[:d], index=mun_votes.index)
        self.Xs[d] = 1  # bias
        self.ks = pd.DataFrame(mun_info.num_valid.values, index=mun_info.index)
        self.params = dict()
        self.d = d + 1
        self.gamma = gamma
        self.n_iter = n_iter

    @staticmethod
    def load_data(path):
        with open(path, 'rb') as f:
            votes, minfo, vinfo = pickle.load(f)
        # Cast to float for weighting at prediction time.
        minfo.num_valid = minfo.num_valid.astype(float)
        return votes, minfo, vinfo

    def init_params(self):
        w = Variable(pt.zeros(self.d))
        w.requires_grad = True
        return w

    def predict(self, res, tol=1e-9):
        vote_id = res[0].vote
        w = self.params.get(vote_id, self.init_params())
        # Filter out new mnicipalities.
        obs, y = list(), list()
        for r in res:
            if r.ogd_id not in self.Xs.index:
                continue
            obs.append(r.ogd_id)
            y.append(r.yes_percent)
        uobs = list(set(self.Xs.index).difference(set(obs)))
        X_train = Variable(
            pt.from_numpy(self.Xs.loc[obs].values.astype(np.float32)))
        y_train = Variable(pt.from_numpy(np.array(y).astype(np.float32)))
        X_pred = Variable(
            pt.from_numpy(self.Xs.loc[uobs].values.astype(np.float32)))
        k_train = Variable(
            pt.from_numpy(
                self.ks.loc[obs].values.flatten().astype(np.float32)))
        # Train.
        losses = list()
        prev_loss = 1e9
        for i in range(self.n_iter):
            inner = X_train @ w
            y_hat = sigmoid(inner)
            loss = (binary_cross_entropy(
                y_hat, y_train, weight=k_train, reduction='sum')
                    / np.sum(k_train.data.numpy()))
            loss.backward()
            w.data -= self.gamma * w.grad.data
            w.grad.data.zero_()
            new_loss = loss.item()
            if prev_loss < new_loss:
                self.gamma /= 2
            if tol is not None and abs(prev_loss - new_loss) < tol:
                break
            losses.append(new_loss)
            prev_loss = new_loss

        # Predict.
        w_pred = w.detach()
        y_pred = sigmoid(X_pred @ w_pred).tolist()

        # Save params for next prediction.
        self.params[vote_id] = w
        return {ogd: pred for ogd, pred in zip(uobs, y_pred)}, losses

    def aggregate(self, obs, uobs, pred):
        aggr, total = dict(), 0
        # Aggregate prediction for unobserved results.
        for res in uobs:
            ogd_id = res.ogd_id
            num_valid = res.num_valid
            if ogd_id in pred:
                aggr[ogd_id] = pred[ogd_id] * num_valid
                total += num_valid
        # Aggregate observed results.
        for res in obs:
            aggr[res.ogd_id] = res.num_valid * res.yes_percent
            total += res.num_valid
        return sum(v for v in aggr.values()) / total
