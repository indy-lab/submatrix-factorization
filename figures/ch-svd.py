import argparse
import matplotlib.pyplot as plt
import pickle

from common import sigconf_settings
from common_projection import svd, plot_svd_with_lang


def plot(args):
    # Set SIGCONF settings.
    sigconf_settings()

    # Load data.
    with open(args.data, 'rb') as f:
        (mun_votes, mun_info) = pickle.load(f)
    print(f'Loaded {args.data}')

    # Filter data.
    lang_avail = (~mun_info.language.isna())
    cant_avail = (~mun_info.canton.isna())
    mun_votes = mun_votes[lang_avail & cant_avail]
    mun_info = mun_info[lang_avail & cant_avail]
    mun_info = mun_info.drop([3805, 3810])
    mun_votes = mun_votes.drop([3805, 3810])

    # Compute SVD embedding.
    embedding = svd(mun_votes.values)

    # Define colors.
    colors = {
        'fr': 'C0',
        'de': 'C1',
        'ro': 'C2',
        'it': 'C3',
        'unknown': 'black'
    }

    # Define labels.
    labels = {
        'fr': 'French',
        'de': 'German',
        'ro': 'Romansh',
        'it': 'Italian'
    }

    # Plot.
    fig, ax = plt.subplots(ncols=1, figsize=(3.6, 1.8))
    plot_svd_with_lang(
        embedding=embedding,
        languages=list(mun_info.language),
        colors=colors,
        labels=labels,
        fig=fig,
        ax=ax
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
