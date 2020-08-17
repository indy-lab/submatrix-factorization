import argparse
import matplotlib.pyplot as plt

from common_results import plot as _plot
from common import sigconf_settings, load


def plot(args):
    # Set SIGCONF settings.
    sigconf_settings()

    # Define type of plot.
    mae = 'mae-outcome'
    dis = 'displacement'

    # Set keys to models.
    models = [
        'Averaging',
        # 'SubSVD (dim=7,l2=0.01)',
        'Logistic SubSVD (dim=7,l2=0.01)'
    ]

    # Define labels.
    labels = [
        'Averaging',
        # r'\textsc{SubSVD-Gaussian}',
        r'\textsc{SubSVD-Categorical}'
    ]

    # Define line styles.
    lines = [
        '-',
        # ':',
        '--',
    ]

    # Define colors.
    colors = [
        'black',
        # 'C3',
        'C3'
    ]

    # Define plot title.
    title = 'German Legislative Election by State'

    # Load data.
    results = load(args.data)

    # Plot.
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(3.6, 2.66))
    _plot(  # MAE.
        results=results,
        metric=mae,
        title=title,
        models=models,
        labels=labels,
        lines=lines,
        colors=colors,
        position='top',
        regions='states',
        ax=ax1,
        fig=fig,
        x_logscale=False
    )
    _plot(  # Displacement.
        results=results,
        metric=dis,
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
