import argparse
import matplotlib.pyplot as plt

from common_results import plot as _plot
from common import sigconf_settings, load


def plot(args):
    # Set SIGCONF settings.
    sigconf_settings()

    # Define type of plot.
    mae = 'mae-outcome'
    acc = 'nat_correct'

    # Set keys to models.
    models = [
        'Weighted Averaging',
        'Matrix Factorization (dim=25,lam_V=0.03,lam_U=31.0)',
        # 'Weighted SubSVD (dim=25,l2=0.1)',
        'Weighted Logistic SubSVD (dim=25,l2=0.1)',
    ]

    # Define labels.
    labels = [
        'Averaging',
        'MF',
        # r'\textsc{SubSVD-Gaussian}',
        r'\textsc{SubSVD-Bernoulli}',
    ]

    # Define line styles.
    lines = [
        '-',
        ':',
        '--',
    ]

    # Define colors.
    colors = [
        'black',
        'black',
        'C3'
    ]

    # Define plot title.
    title = 'Swiss Referenda'

    # Load data.
    results = load(args.data)

    # Plot.
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(3.5, 2.66))
    _plot(  # MAE.
        results=results,
        metric=mae,
        title=title,
        models=models,
        labels=labels,
        lines=lines,
        colors=colors,
        regions='municipalities',
        position='top',
        ax=ax1,
        fig=fig,
        x_logscale=True
    )
    _plot(  # Accuracy.
        results=results,
        metric=acc,
        title=title,
        models=models,
        labels=labels,
        lines=lines,
        colors=colors,
        position='bottom',
        regions='municipalities',
        ax=ax2,
        fig=fig,
        x_logscale=True
    )
    fig.tight_layout()
    plt.savefig(args.fig)
    print(f'Figure saved to {args.fig}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='Path to data.')
    parser.add_argument('fig', help='Path to save fig.')
    args = parser.parse_args()
    plot(args)
