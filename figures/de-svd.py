import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle

from benchmarks.datasets import GermanParliamentRegion
from common import sigconf_settings
from common_projection import svd, min_max_scale
from matplotlib import rcParams


def plot_nth_scatter(embedding, n, ax, colorixs, colors, parties, title,
                     print_names=True):
    # Rescale embeddings.
    embedding = min_max_scale(embedding, axis=0)

    # Get items to plot.
    cols = [colors[c[-n]] for c in colorixs]
    labels = [parties[c[-n]] for c in colorixs]
    xs, ys = embedding[:, 0], embedding[:, 1]
    for x, y, c, label in zip(xs, ys, cols, labels):
        ax.scatter(
            x=x,
            y=y,
            c=c,
            label=label,
            s=0.8,
            lw=0.8,
        )
    # Display Berlin on the map.
    if n == 1:  # Get plot with first party.
        arrow = dict(width=0.1, headwidth=4, headlength=4, facecolor='black')
        ax.annotate('Berlin', xy=(0.3, 0.59), xytext=(0.45, 0.59),
                    arrowprops=arrow)
    # Plot "Historical East/West" on plot with third parties.
    elif n == 3:  # Get plot with third party.
        lower, upper = [-0.05, 0.2], [1, 0.8]
        ax.plot([lower[0], upper[0]], [lower[1], upper[1]],
                c='black', lw=1, linestyle='--')
        ax.text(-0.08, 0.23, 'Historical East/West', rotation=30, fontsize=8)
    # Set title.
    ax.set_title(title)
    # Remove legend border.
    rcParams.update({'legend.frameon': False})
    # Remove ticks.
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    # Sort legend by alphabetical order.
    handles, labels = ax.get_legend_handles_labels()
    # Keep only on label.
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels))
              if l not in labels[:i]]
    handles, labels = zip(*sorted(unique, key=lambda t: t[1]))
    # Configure legend.
    if n == 1:
        lgnd = ax.legend(
            handles,
            labels,
            loc='best',
            handletextpad=0.,
            labelspacing=0.2
        )
    elif n == 3:
        lgnd = ax.legend(
            handles,
            labels,
            loc='lower right',
            bbox_to_anchor=(1.025, -0.05),
            handletextpad=0.,
            labelspacing=0.2
        )
    # Set size of points in legend (must be done after legend is set).
    for handle in lgnd.legendHandles:
        handle.set_sizes([10])


def main(args):
    # Set SIGCONF settings.
    sigconf_settings()

    # Load data.
    ds = GermanParliamentRegion(args.data)
    data = ds.M.reshape(ds.shape[0], -1)
    print(f'Loaded {args.data}')

    parties = ['SPD', 'CDU/CSU', 'Greens', 'FDP', 'Left']
    colors = ['C3', 'C0', 'C2', 'C1', 'C6']
    colors = [f'C{i}' for i in range(len(parties))]
    colorixs = np.argsort(ds.M.mean(axis=1), axis=1)

    # Get SVD embedding.
    if args.embedding is not None:
        with open(args.embedding, 'rb') as f:
            (embedding, colorixs) = pickle.load(f)
        print(f'Loaded embedding from {args.embedding}')
    else:
        embedding = svd(data)
        path = f'{args.data}/de-embedding.pkl'
        with open(path, 'wb') as f:
            pickle.dump((embedding, colorixs), f)
        print(f'Saved embedding in {path}')

    # Plot.
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(3.6, 2))
    plot_nth_scatter(
        embedding=embedding,
        n=1,
        ax=ax1,
        colorixs=colorixs,
        colors=colors,
        parties=parties,
        title='Coloring by First Party',
        print_names=True
    )
    plot_nth_scatter(
        embedding=embedding,
        n=3,
        ax=ax2,
        colorixs=colorixs,
        colors=colors,
        parties=parties,
        title='Coloring by Third Party',
        print_names=True
    )
    fig.tight_layout()
    plt.savefig(args.fig)
    print(f'Figure saved to {args.fig}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='Path to data.')
    parser.add_argument('fig', help='Path to save fig.')
    parser.add_argument('--embedding', help='Path to embedding.')
    args = parser.parse_args()
    main(args)
