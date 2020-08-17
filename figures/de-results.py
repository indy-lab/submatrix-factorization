import argparse
import matplotlib.pyplot as plt

from common_results import plot as _plot
from common import sigconf_settings, load


def plot(args):
    # Set SIGCONF settings.
    sigconf_settings()

    # Setup figure layout.
    gridspec = dict(wspace=0.1, bottom=0.2, right=0.95)
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(3.6, 2.66),
        sharex=False,
        sharey='row',
        gridspec_kw=gridspec
    )

    # Plot district and state results.
    plot_state(args, fig, axes[0, 0], axes[1, 0])
    plot_district(args, fig, axes[0, 1], axes[1, 1])

    # Plot and save figure.
    plt.savefig(args.fig)
    print(f'Figure saved to {args.fig}')


def plot_state(args, fig, ax1, ax2):
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
        r'\textsc{SubSVD-Categ.}'
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
    title = 'German Election by State'

    # Load data.
    results = load(args.state)

    # Plot by states.
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
        x_logscale=False,
        legend=False
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
        x_logscale=False,
    )


def plot_district(args, fig, ax1, ax2):
    # Define type of plot.
    mae = 'mae-outcome'
    dis = 'displacement'

    # Set keys to models.
    models = [
        'Averaging',
        # 'SubSVD (dim=11,l2=0.01)',
        'Logistic SubSVD (dim=11,l2=0.01)'
    ]

    # Define labels.
    labels = [
        'Averaging',
        # r'\textsc{SubSVD-Gaussian}',
        r'\textsc{SubSVD-Categ.}'
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
    title = 'German Election by District'

    # Load data.
    results = load(args.district)

    # Plot by districts.
    _plot(  # MAE.
        results=results,
        metric=mae,
        title=title,
        models=models,
        labels=labels,
        lines=lines,
        colors=colors,
        position='top',
        regions='districts',
        ax=ax1,
        fig=fig,
        ylim=(-0.0025, 0.06),
        x_logscale=True,
        ylabel=False
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
        regions='districts',
        ax=ax2,
        fig=fig,
        ylim=(-0.05, 1.12),
        x_logscale=True,
        ylabel=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('state', help='Path to state data.')
    parser.add_argument('district', help='Path to district data.')
    parser.add_argument('fig', help='Path to save fig.')
    args = parser.parse_args()
    plot(args)
