import pickle
import json
import os
import numpy as np
import pandas as pd


class Column:

    """A column of the data matrix whose entries arrive sequentially.
    For instance, this can be a new vote.
    :param column: Array of values in the column.
    :param n_orders: Number of random reveal orders.
    """

    def __init__(self, values, n_orders=5):
        self.values = values
        self._n_orders = n_orders

    @property
    def reveal_orders(self):
        """Generate random reveal orders."""
        order = np.arange(len(self.values))
        for i in range(self._n_orders):
            np.random.shuffle(order)
            yield order


class Dataset:

    """Class to manage a dataset.
    :param data_dir: Absolute path to directory of dataset.
    :param M: The full matrix M.
    :param weights: Weighting for each sample (e.g., regional population).
    :param outcomes: True outcomes associated to a column.
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.M = None
        self.weights = None
        self.outcomes = None
        self.n_parties = None
        self.load()
        assert self.M is not None
        assert self.weights is not None
        assert self.n_parties is not None

    def tasks(self, n_obs, n_orders, v_max=None):
        """Generate sequence of tasks, i.e., a sub-matrix and a new column.
        The sub-matrix is used to make predictions on the partially observed
        new column.
        :param n_obs: Initial number of columns in sub-matrix.
        :param n_orders: Number of random reveal orders for the new column.
        :param v_max: Maximum number of votes to predict, defaults to all
        """
        if v_max is None:
            v = self.M.shape[1]
        else:
            v = v_max
        for i in range(n_obs, v):
            subM = self.M[:, :i]
            m = self.M[:, i]
            yield subM, Column(m, n_orders), self.weights

    def load(self):
        raise NotImplementedError

    @property
    def shape(self):
        return self.M.shape


class SwissVote(Dataset):

    def load(self):
        with open(self.data_dir + 'munvoteinfo.pkl', 'rb') as f:
            votes, minfo, vinfo = pickle.load(f)
        # Cast to float for weighting at prediction time.
        minfo.num_valid = minfo.num_valid.astype(float)
        self.M = votes.to_numpy()
        self.weights = minfo.num_valid.to_numpy()
        self.outcomes = vinfo.yes_percent.to_numpy()
        self.n_parties = 2


class USPresidencyElection(Dataset):

    def load(self):
        frame = pd.read_csv(self.data_dir + '1976-2016-us-president.tab', sep='\t')
        ixs = (frame.party == 'democrat') | (frame.party == 'republican')
        frame = frame[ixs.values]
        grp = frame.groupby('year')
        votes = list()
        years = list()
        states = set(grp.get_group(1976).state)
        for year in set(frame.year):
            fgrp = grp.get_group(year)
            fgrp = fgrp[fgrp.party == 'democrat']
            states = states.intersection(set(fgrp.state))
        states = list(states)
        for year in set(frame.year):
            years.append(year)
            fgrp = grp.get_group(year)
            fsubgrp = fgrp.groupby('state').sum()
            total_votes = fsubgrp.candidatevotes
            fgrp = fgrp[fgrp.party == 'democrat']
            fgrp = fgrp.groupby('state').sum()
            fgrp = fgrp.loc[states]
            ps = (fgrp.candidatevotes / total_votes[fgrp.index]).values
            votes.append(ps)
            if year == 2016 or year == '2016':
                # weighting by 2016 election  (public type)
                self.weights = total_votes[fgrp.index].values
        # aggregated democrat votes in the 50 american states
        # fraction of votes for democrat + republican for all candidates
        # re-order by years correctly so that last vote is 2016
        self.M = np.stack(votes)[np.argsort(years)].T
        self.years = sorted(years)
        self.states = states
        self.n_parties = 2

        with open(self.data_dir + 'electoral-votes.json', 'r') as f:
            maps = [json.loads(l) for l in f.readlines()]
        mapper = {d['state']: d['votes'] for d in maps}
        self.electoral_weights = np.array([mapper[state] for state in states])


class GermanParliamentRegion(Dataset):

    def load(self):
        docs = list()
        years = list()
        skiprows = [0, 1, 2, 3, 4, 5, 7]
        for f in os.listdir(self.data_dir + '/german_parliament_NUTS3/'):
            if 'NUTS3' not in f:
                continue
            d = pd.read_excel(self.data_dir + '/german_parliament_NUTS3/' + f,
                              skiprows=skiprows, index_col=0)
            if '1990' in f:
                d['Grüne'] += d['B90/GRÜ']
                d.drop(columns='B90/GRÜ', inplace=True)
                d.rename(columns={'Grüne': 'B90/GRÜ'}, inplace=True)
            if '2005' in f:
                d.rename(columns={'Linke': 'PDS'}, inplace=True)
            docs.append(d)
            years.append(f.split('_')[-1][:4])

        base_regions = set(docs[0].index)
        for d in docs:
            base_regions = base_regions.intersection(set(d.index))
        base_regions = list(base_regions - {'Eisenach, Kreisfreie Stadt'})
        base_parties = ['SPD', 'CDU/CSU', 'B90/GRÜ', 'FDP', 'Linke']

        for i, d in enumerate(docs):
            d.rename(columns={'PDS': 'Linke'}, inplace=True)
            docs[i] = d.loc[base_regions, base_parties]

        # write into n_votes x n_regions x n_partes tensor and sort by year
        res_tensor = np.stack([d.to_numpy() for d in docs]).astype(np.float)
        res_tensor = res_tensor[np.argsort(years)]
        self.M = res_tensor / res_tensor.sum(axis=-1)[:, :, np.newaxis]
        self.M = self.M.transpose((1, 0, 2))
        self.parties = base_parties
        self.regions = base_regions
        self.weights = np.ones(self.M.shape[0])
        self.n_parties = len(base_parties)


class GermanParliamentState(Dataset):

    def load(self):
        docs = list()
        years = list()
        skiprows = [0, 1, 2, 3, 4, 5, 7]
        for f in os.listdir(self.data_dir + '/german_parliament_NUTS1/'):
            if 'NUTS1' not in f:
                continue
            d = pd.read_excel(self.data_dir + '/german_parliament_NUTS1/' + f,
                             skiprows=skiprows, index_col=0)
            if '1990' in f:
                d['Grüne'] += d['B90/GRÜ']
                d.drop(columns='B90/GRÜ', inplace=True)
                d.rename(columns={'Grüne': 'B90/GRÜ'}, inplace=True)
            if '2005' in f:
                d.rename(columns={'Linke': 'PDS'}, inplace=True)
            if '2009' in f:
                d['CDU'] += d['CSU']
                d.drop(columns='CSU', inplace=True)
                ren = {'DIE LINKE': 'PDS', 'GRÜNE': 'B90/GRÜ',
                       'CDU': 'CDU/CSU'}
                d.rename(columns=ren, inplace=True)
            d.index = d.index.map(str.lower)
            docs.append(d)

            year = int(f.split('_')[-1][:4])
            if f[-5] == 'b':
                # make year of second elections (2009b) to 2010
                year += 1
            years.append(year)

        base_regions = set(docs[0].index)
        for d in docs:
            base_regions = base_regions.intersection(set(d.index))
        base_parties = ['SPD', 'CDU/CSU', 'B90/GRÜ', 'FDP', 'Linke']

        for i, d in enumerate(docs):
            d.rename(columns={'PDS': 'Linke'}, inplace=True)
            docs[i] = d.loc[base_regions, base_parties]

        # write into n_votes x n_regions x n_partes tensor and sort by year
        res_tensor = np.stack([d.to_numpy() for d in docs]).astype(np.float)
        res_tensor = res_tensor[np.argsort(years)]
        self.M = res_tensor / res_tensor.sum(axis=-1)[:, :, np.newaxis]
        self.M = self.M.transpose((1, 0, 2))
        self.parties = base_parties
        self.regions = base_regions
        self.weights = np.ones(self.M.shape[0])
        self.n_parties = len(base_parties)

