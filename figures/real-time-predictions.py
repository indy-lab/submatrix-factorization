import argparse
import matplotlib.pyplot as plt
import numpy as np

from common import sigconf_settings, load, format_time


# Keep for reference.
# VOTES = [
#     6270,  # RFFA.
#     6280,  # Schengen.
#     6290,  # Affordable houses.
#     6300,  # Discrimination.
# ]
DATE2VOTES = {
    '2019-05-19': [6270, 6280],
    '2020-02-09': [6290, 6300]
}
VOTE2IDX = {
    6290: 0,
    6300: 1
}


def plot(preds, ax, labels, colors, linestyles, title, pos):
    # Get series.
    ypred = [p['ypred'] for p in preds]
    count = [p['count'] for p in preds]
    ys = np.array([count, ypred])
    x = np.arange(len(ypred))
    ts = [p['timestamp'] for p in preds]

    # Plot prediction and averaging.
    for y, label, color, linestyle in zip(ys, labels, colors, linestyles):
        ax.plot(
            x,
            y * 100,
            linewidth=2,
            label=label,
            linestyle=linestyle,
            color=color,
        )
    ax.set_ylabel('Outcome [%]')
    xticks = np.arange(0, len(ts), 10)
    ax.set_xticks(xticks)
    if pos == 'top':
        ax.xaxis.set_major_formatter(plt.NullFormatter())
        ax.set_ylim([36, 44])
    elif pos == 'bottom':
        xlabels = [format_time(ts[t]) for t in xticks]
        ax.set_xticklabels(xlabels, fontsize=6)
        ax.set_xlabel('Time [pm]')
        ax.set_ylim([58, 64])

    # Plot counting progress.
    counting = np.array([p['counting'] for p in preds]) * 100
    ax2 = ax.twinx()
    color = 'C0'
    ax2.plot(
        x,
        counting,
        linewidth=2,
        color=color,
        label='Counted Ballots'
    )
    # Set counting yticks.
    ax2.set_ylabel('Counting [%]', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0, 100])
    ax2.set_yticks(np.linspace(
        ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax.get_yticks())))
    yticks = ax2.get_yticks()
    ax2.set_yticklabels([f'{int(t)}' for t in yticks])

    # Fix z-order.
    ax.set_zorder(ax2.get_zorder()+1)
    ax.patch.set_visible(False)

    # Fake a vertical grid as ax2 doesn't have one (it is a twin).
    for xt in xticks:
        ax2.axvline(xt, zorder=-1, c='#b0b0b0', linewidth=0.8)

    # Set plot config.
    ax.set_title(title)
    ax2.grid(which='both')
    # Show legend on bottom plot.
    if pos == 'bottom':
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            handles + handles2,
            labels + labels2,
            loc='best',
            labelspacing=0.3
        )


def main(args):
    # Set SIGCONF settings.
    sigconf_settings()

    # Load predictions.
    data = load(args.data)

    # Define labels.
    labels = [
        'Averaging',
        r'\textsc{SubSVD-Bernoulli}',
    ]

    # Define colors.
    colors = [
        'black',
        'C3',
    ]

    # Define line styles.
    linestyles = [
        '-',
        # '--',
        '-'
    ]

    # Set plot positions.
    positions = ['top', 'bottom']
    # Set titles.
    titles = ['Affordable Houses', 'Ban on Homophobia']

    fig, axes = plt.subplots(nrows=2, figsize=(3.6, 2.8))
    for vote, preds in data.items():
        if vote not in DATE2VOTES['2020-02-09']:
            continue
        i = VOTE2IDX[vote]
        plot(preds, axes[i], labels, colors, linestyles,
             titles[i], positions[i])

    fig.tight_layout()
    plt.savefig(args.fig)
    print(f'Figure saved to {args.fig}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='Path to data.')
    parser.add_argument('fig', help='Path to save fig.')
    args = parser.parse_args()
    main(args)
