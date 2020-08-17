import argparse
import numpy as np
import pickle

from benchmarks.datasets import (SwissVote, USPresidencyElection,
                                 GermanParliamentState, GermanParliamentRegion)
from benchmarks.metrics import rmse, mae, weighted_mae
from benchmarks.models import (WeightedAveraging, Averaging, SubSVD,
                               WeightedSubSVD, LogisticSubSVD,
                               WeightedLogisticSubSVD, MatrixFactorisation,
                               TensorSubSVD, LogisticTensorSubSVD)
from math import ceil


def get_models(train, dataset):
    if not train:
        """TEST MODELS"""
        if dataset == 'US':
            models = [
                (WeightedAveraging, dict()),
                (Averaging, dict()),
                (MatrixFactorisation, dict(n_dim=5, lam_V=0.01, lam_U=0.01)),
                (WeightedSubSVD, dict(n_dim=5, l2_reg=0.001)),
                (WeightedLogisticSubSVD, dict(n_dim=5, l2_reg=0.001)),
           ]
        elif dataset == 'CH':
            models = [
                (WeightedAveraging, dict()),
                (MatrixFactorisation, dict(n_dim=25, lam_V=0.03, lam_U=31.)),
                (WeightedSubSVD, dict(n_dim=25, l2_reg=0.1)),
                (WeightedLogisticSubSVD, dict(n_dim=25, l2_reg=0.1)),
            ]
        elif dataset == 'DEState':
            models = [
                (Averaging, dict()),
                (TensorSubSVD, dict(l2_reg=0.01, n_dim=7)),
                (LogisticTensorSubSVD, dict(l2_reg=0.01, n_dim=7)),
            ]
        elif datset == 'DELocal':
            models = [
                (Averaging, dict()),
                (TensorSubSVD, dict(l2_reg=0.01, n_dim=11)),
                (LogisticTensorSubSVD, dict(l2_reg=0.01, n_dim=11)),
            ]
        return models

    """TRAIN MODELS"""
    if dataset in ['US', 'CH']:
        # binary outcomes
        models = [
            (WeightedAveraging, dict()),
            (Averaging, dict())
        ]
        # training models
        l2_regs = [1e-4, 1e-3, 1e-2, 1e-1]
        if dataset == 'US':
            n_dims = [3, 5, 7]
        else:
            n_dims = [10, 50, 100, 250]
        for l2_reg in l2_regs:
            for n_dim in n_dims:
                param = dict(n_dim=n_dim, l2_reg=l2_reg)
                models.append((SubSVD, param))
                models.append((WeightedSubSVD, param))
                models.append((LogisticSubSVD, param))
                models.append((WeightedLogisticSubSVD, param))

        if dataset == 'US':
            # for CH use optimal parameters according to Etter (Table III)
            lams_U = [1e-2, 1e0, 1e2]
            lams_V = lams_U
            for n_dim in n_dims:
                for lam_V in lams_V:
                    for lam_U in lams_U:
                        param = dict(n_dim=n_dim, lam_U=lam_U, lam_V=lam_V)
                        models.append((MatrixFactorisation, param))

    else:
        l2_regs = [1e-3, 1e-2, 1e-1]
        n_dims = [3, 7, 11]
        models = [(Averaging, dict())]
        for n_dim in n_dims:
            for l2_reg in l2_regs:
                params = dict(l2_reg=l2_reg, n_dim=n_dim)
                models.append((TensorSubSVD, params))
                models.append((LogisticTensorSubSVD, params))

    return models


def run(args):
    # Initialize dataset.
    if args.dataset == 'US':
        dataset = USPresidencyElection(args.data_dir)
    elif args.dataset == 'CH':
        dataset = SwissVote(args.data_dir)
    elif args.dataset == 'DEState':
        dataset = GermanParliamentState(args.data_dir)
    elif args.dataset == 'DELocal':
        dataset = GermanParliamentRegion(args.data_dir)
    n_parties = dataset.n_parties

    def accurate(nat_pred, nat_true):
        return 1. if (nat_pred <= 0.5 and nat_true <= 0.5) \
                or (nat_pred >= 0.5 and nat_true >= 0.5) else 0.

    def agreement(pred_out, outcome):
        ipred, itrue = np.argsort(pred_out), np.argsort(outcome)
        return np.sum(ipred == itrue) / len(pred_out)

    def displacement(pred_out, outcome):
        ipred, itrue = np.argsort(pred_out), np.argsort(outcome)
        disp = np.sum(np.abs(ipred - itrue)) / len(ipred)
        max_avg_disp = len(ipred) / 2
        disp_acc = (max_avg_disp - disp) / max_avg_disp
        return disp_acc

    def evaluate():
        """Evaluate performance."""
        # Compute performance on missing entries.  - Regional performance
        results['mae-col-unobs'][i, j, k, h] = mae(pred[m_u], m[m_u])
        results['wmae-col-unobs'][i, j, k, h] = weighted_mae(pred[m_u], m[m_u], weights[m_u])
        # Compute performance on final outcome.  - national performance
        W = w_aggr.sum()
        pred_out = w_aggr.dot(pred_aggr) / W
        outcome = w_aggr.dot(m_aggr) / W
        results['mae-outcome'][i, j, k, h] = mae(pred_out, outcome)
        if n_parties > 2:
            results['nat_agree'][i, j, k, h] = agreement(pred_out, outcome)
            results['displacement'][i, j, k, h] = displacement(pred_out, outcome)
        else:
            results['nat_correct'][i, j, k, h] = accurate(pred_out, outcome)

    # Set the seed.
    np.random.seed(args.seed)
    # select model group
    MODELS = get_models(args.models=='train', args.dataset)
    # Define results array.
    axes = ['models', 'tasks', 'orders', 'observations']
    metrics = ['mae-col-unobs', 'mae-outcome', 'wmae-col-unobs']
    if n_parties > 2:
        metrics += ['nat_agree', 'displacement']
    else:
        metrics += ['nat_correct']

    m, v = dataset.shape[:2]
    n_tasks = args.v_max - args.n_obs
    if args.logscale is True:
        exp = np.log10(m-1)
        x_axis = np.unique(np.logspace(0, exp, args.n_quantiles).round().astype(int))
    else:
        x_axis = np.unique(np.linspace(1, m-1, args.n_quantiles).astype(int))
    results = {metric: np.zeros((len(MODELS), n_tasks, args.n_orders, len(x_axis)))
               for metric in metrics}

    # For each (predictive) task.
    tasks = dataset.tasks(args.n_obs, args.n_orders, v_max=args.v_max)
    for j, (M, column, weights) in enumerate(tasks):
        print(f'##### TASK {j+1}')
        # For each reveal order on the column.
        for k, order in enumerate(column.reveal_orders):
            print(f'##### ORDER {k+1}')
            # For each model.
            init_models = list()
            for i, (Model, kwargs) in enumerate(MODELS):
                model = Model(M, weights, **kwargs)
                # For each revealed result.
                for h, l in enumerate(x_axis):
                    # Set observed and unobserved indices.
                    m_o = order[:l]
                    m_u = order[l:]
                    # Predict missing entries.
                    m = column.values
                    pred = model.fit_predict(m, m_o, m_u)
                    # Aggregate observed and unobserved values.
                    pred_aggr = np.concatenate([pred[m_u], m[m_o]])
                    m_aggr = np.concatenate([m[m_u], m[m_o]])
                    w_aggr = np.concatenate([weights[m_u], weights[m_o]])
                    # Evaluate performance.
                    evaluate()
                init_models.append(model)

    # Save results.
    models = [m.name for m in init_models]
    with open(f'{args.name}.pkl', 'wb') as f:
        pickle.dump({'results': results, 'axes': axes, 'models': models,
                     'x': x_axis, 'n_parties': dataset.n_parties}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Model selection experiment with result saving.')

    parser.add_argument('--name', type=str, required=True,
                        help='Name of experiment.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to directory of data.')
    parser.add_argument('--n_obs', type=int, default=10,
                        help='Initial number of columns in sub-matrix.')
    parser.add_argument('--v_max', type=int, default=None,
                        help='Last vote to predict.' +
                        'Will predict v_max - n_obs')
    parser.add_argument('--n_orders', type=int, default=5,
                        help='Number of random reveal orders.')
    parser.add_argument('--dataset', type=str, choices=['CH', 'US', 'DELocal', 'DEState'],
                        help='Either Swiss referenda or US elections. For Parties DELocal/State')
    parser.add_argument('--models', type=str, choices=['train', 'test'],
                        help='Training or testing models.')
    parser.add_argument('--n_quantiles', type=int, default=10,
                        help='Quantiles in a vote where to make predictions.')
    parser.add_argument('--logscale', action='store_true')
    parser.add_argument('--seed', type=int, default=0,
                        help='Set the seed for reproducibility.')
    run(parser.parse_args())
