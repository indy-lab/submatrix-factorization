import pickle
import numpy as np


def make_list(args):
    def performance(metric):
        # Keep non zero predictions only.
        val = results[metric]
        # Aggregate mean results across votes and across reveal orders.
        agg = val.mean(axis=(1, 2, 3))
        return agg

    with open(f'{args.name}.pkl', 'rb') as f:
        res = pickle.load(f)

    mname = lambda x: x.split('(')[0]
    results, models = res['results'], res['models']
    perfs_out = performance('mae-outcome')
    model_names = set([mname(m) for m in models])
    models = np.array(models)
    print('----------MAE PERFORMANCE (Top to Bottom)----------')
    for m in model_names:
        ixs = [i for i in range(len(models)) if m == mname(models[i])]
        print('FOR model type ' + m)
        # print model configurations
        print(models[ixs][np.argsort(perfs_out[ixs])])
        # print model performances
        print(np.sort(perfs_out[ixs]))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate and sort models')
    parser.add_argument('--name', type=str, required=True)
    make_list(parser.parse_args())
