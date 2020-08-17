from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE
from matplotlib.patches import Ellipse


LANGS = ['de', 'fr', 'it', 'ro']


def svd(M, skip=0):
    M = M - M.mean(axis=0)
    U, S, V = np.linalg.svd(M)
    E = M.dot(V.T[:, skip:skip+2])
    return E


def tsne(M, seed=0):
    ts = TSNE(random_state=seed)
    E = ts.fit_transform(M)
    return E


def min_max_scale(M, axis=0):
    m_max = M.max(axis=axis)
    m_min = M.min(axis=axis)
    return (M - m_min) / (m_max - m_min)


def lang_color(lang):
    lang_color = {
        'fr': 'purple',
        'de': 'blue',
        'ro': 'green',
        'it': 'red',
        'unknown': 'black'
    }
    if lang not in lang_color:
        return 'yellow'
    return lang_color[lang]


def get_canton_coloring(cantons):
    n = len(cantons)
    if None in cantons:
        n -= 1
        cantons.remove(None)
    cantons = list(cantons)

    def color_canton(cant):
        if cant is None:
            return 1
        else:
            return cantons.index(cant) / (n+1)

    return color_canton


def plot_svd_with_lang(embedding, languages, colors, labels, fig, ax):
    assert len(embedding) == len(languages)

    # Rescale embeddings.
    embedding = min_max_scale(embedding, axis=0)
    # Get panguages.
    langs = set(languages)
    # Plot municipalities in vote space.
    for lang in langs:
        ixs = [i for i, la in enumerate(languages) if la == lang]
        ax.scatter(
            x=-embedding[ixs, 0],
            y=embedding[ixs, 1],
            alpha=1,
            c=colors[lang],
            s=0.8,
            linewidths=0.8,
            label=labels[lang])
    # Plot Röstigraben.
    lower, upper = [-0.75, -0.05], [-0.05, 1]
    ax.plot([lower[0], upper[0]], [lower[1], upper[1]],
            c='black', lw=1, linestyle='--')
    ax.text(-0.78, -0.04, '``Röstigraben"', rotation=33, fontsize=8)
    # Remove legend border.
    rcParams.update({'legend.frameon': False})
    # Remove ticks.
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    # Sort legend by alphabetical order.
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    # Configure legend.
    lgnd = ax.legend(
        handles,
        labels,
        loc='best',
        # bbox_to_anchor=(-0.05, -0.02),
        handletextpad=0.,
        labelspacing=0.2
    )
    # Set size of points in legend (must be done after legend is set).
    for handle in lgnd.legendHandles:
        handle.set_sizes([10])
    # Set title.
    # ax.set_title('Coloring by Language')


def plot_embedding_with_lang(embedding, languages, colors, labels, fig, ax):
    assert len(embedding) == len(languages)

    # Rescale embeddings.
    embedding = min_max_scale(embedding, axis=0)
    # Get panguages.
    langs = set(languages)

    for lang in langs:
        ixs = [i for i, la in enumerate(languages) if la == lang]
        ax.scatter(
            x=-embedding[ixs, 0],
            y=-embedding[ixs, 1],
            alpha=1,
            c=colors[lang],
            s=0.8,
            linewidths=0.8,
            label=labels[lang])
    # Highlight Wallis.
    wallis = Ellipse(xy=(-0.56, -0.06), width=0.4, height=0.2,
                     color='k', fill=False, linestyle='--', linewidth=0.5)
    ax.add_patch(wallis)
    ax.text(-0.34, -0.08, 'Wallis', fontsize=8)
    # Highlight Ticino.
    ticino = plt.Circle((-0.512, -0.945), 0.09,
                        linestyle='--', linewidth=0.5, fill=False)
    ax.add_artist(ticino)
    ax.text(-0.4, -0.97, 'Ticino', fontsize=8)
    # Remove legend border.
    rcParams.update({'legend.frameon': False})
    # Remove ticks.
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    # Sort legend by alphabetical order.
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    # Configure legend
    lgnd = ax.legend(
        handles,
        labels,
        loc='lower left',
        bbox_to_anchor=(-0.05, -0.02),
        handletextpad=0.,
        labelspacing=0.2
    )
    # Set size of points in legend.
    for handle in lgnd.legendHandles:
        handle.set_sizes([10])
    # Set title.
    ax.set_title('Coloring by Language')


def plot_embedding_with_canton_and_lang(
        embedding, cantons, languages, fig, ax):
    assert len(embedding) == len(cantons)

    # Rescale embeddings.
    embedding = min_max_scale(embedding, axis=0)
    unique_cantons = set(cantons)
    cant_color = get_canton_coloring(unique_cantons)

    labeled = {c: False for c in unique_cantons}
    for lang in LANGS:
        lixs = [i for i, la in enumerate(languages) if la == lang]
        for canton in unique_cantons:
            ixs = [i for i in lixs if cantons[i] == canton]
            c = np.array([plt.cm.viridis(cant_color(canton))])
            label = canton if not labeled[canton] else None
            ax.scatter(
                x=-embedding[ixs, 0],  # Reverse (x, y) for display purposes.
                y=-embedding[ixs, 1],
                alpha=1,
                c=c,
                vmin=0,
                vmax=1,
                s=0.8,
                linewidths=0.8,
                label=label,
                # marker=lang_markers[lang]
            )
            labeled[canton] = True
    # Remove legend border.
    rcParams.update({'legend.frameon': False})
    # Remove ticks.
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    # Set title.
    ax.set_title('Coloring by Canton')
