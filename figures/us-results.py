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
        'Matrix Factorization (dim=5,lam_V=0.01,lam_U=0.01)',
        # 'Weighted SubSVD (dim=5,l2=0.001)',
        'Weighted Logistic SubSVD (dim=5,l2=0.001)',
        # 'SubSVD (dim=5,l2=0.001)',
        # 'Logistic SubSVD (dim=5,l2=0.001)',
    ]

    # Define labels.
    labels = [
        'Averaging',
        'MF',
        # r'\textsc{SubSVD-Gaussian}',
        r'\textsc{SubSVD-Bernoulli}',
        # r'\textsc{UWSubSVD-Gaussian}',
        # r'\textsc{UWSubSVD-Bernoulli}',
    ]

    # Define line styles.
    lines = [
        '-',
        ':',
        '--',
        # '-.',
        # '-',
        # '-',
    ]

    # Define colors.
    colors = [
        'black',
        'black',
        'C3',
        # 'C1',
        # 'C0',
        # 'C4',
    ]

    # Define plot title.
    title = 'U.S. Presidential Election 2016'

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
        regions='states',
        position='top',
        ax=ax1,
        fig=fig,
        x_logscale=False
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
        regions='states',
        ax=ax2,
        fig=fig,
        x_logscale=False
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
