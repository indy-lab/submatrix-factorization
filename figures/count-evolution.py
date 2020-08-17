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


def plot_evolution(ys, ts, ax, labels, linestyles):
    # Find the time step where all votes are counted.
    lmax, ysmean = 0, dict()
    for date, votes in DATE2VOTES.items():
        # Average the valid votes for each vote.
        y = np.mean([ys[vote] for vote in votes], axis=0)
        ysmean[date] = y
        if len(y) > lmax:
            lmax = len(y)

    # Plot.
    for date, y in ysmean.items():
        # Pad smallest time series.
        if len(y) < lmax:
            yl = list(y)
            for _ in range(lmax - len(y)):
                yl.append(yl[-1])
            y = np.array(yl)
        x = np.arange(len(y))
        ax.plot(
            x,
            y,
            linewidth=2,
            label=labels[date],
            linestyle=linestyles[date],
            # color='black'
        )

    # Set title.
    # ax.set_title('Evolution of number of voters over time')
    # Set ticks.
    yticks = ax.get_yticks()
    ax.set_yticklabels([t/1e6 for t in yticks])
    tmax = list()
    for t in ts.values():
        if len(t) > len(tmax):
            tmax = t

    xticks = np.arange(0, len(tmax), 10)
    ax.set_xticks(xticks)
    xlabels = [format_time(tmax[t]) for t in xticks]
    ax.set_xticklabels(xlabels)
    ax.set_xlabel('Time [pm]')

    # xticks = [0] + list(range(10, imax, 10)) + [imax-1]
    # xticks = np.arange(0, imax, 10)
    # ax.set_xticks(xticks)
    # xlabels = [format_time(ts[t]) for t in xticks]
    # ax.set_xticklabels(xlabels)

    # Set labels.
    ax.set_ylabel('Ballot papers [in million]')
    ax.set_xlabel('Time [pm]')
    # Set plot config.
    ax.legend()
    ax.grid(which='both')


def plot(args):
    # Set SIGCONF settings.
    sigconf_settings()

    # Load data.
    data = load(args.data)
    valid_votes = data['valid_votes']

    # Get time series.
    ts = {6270: list(), 6280: list(), 6290: list(), 6300: list()}
    ys = {6270: list(), 6280: list(), 6290: list(), 6300: list()}
    for vote, counts in valid_votes.items():
        for t, count in counts.items():
            ys[vote].append(count)
            ts[vote].append(t)

    # Labels.
    labels = {
        '2019-05-19': 'May 19, 2019',
        '2020-02-09': 'Feb 2, 2020',
    }

    # Line style.
    linestyles = {
        '2019-05-19': '-',
        '2020-02-09': '-',
    }

    # Plot.
    fig, ax = plt.subplots(figsize=(3.6, 1.8))

    plot_evolution(ys, ts, ax, labels, linestyles)

    fig.tight_layout()
    plt.savefig(args.fig)
    print(f'Figure saved to {args.fig}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='Path to data.')
    parser.add_argument('fig', help='Path to save fig.')
    args = parser.parse_args()
    plot(args)
