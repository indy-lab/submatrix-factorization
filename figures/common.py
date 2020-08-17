import matplotlib
import pickle
import re

from matplotlib.backends.backend_pgf import FigureCanvasPgf

SIGCONF_RCPARAMS = {
    "figure.autolayout": True,          # Make sure the figure is neat & tight.
    "figure.figsize": (7.0, 3.0),       # Column width: 3.333in
                                        # Space between columns: 0.333in
    "figure.dpi": 150,                  # Displays figures nicely in notebooks.
    "axes.linewidth": 0.5,              # Matplotlib's current default is 0.8.
    "hatch.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    "ytick.major.width": 0.5,
    "ytick.minor.width": 0.5,
    "text.usetex": True,                # Use LaTeX to write all text
    "font.family": "serif",             # Use serif rather than sans-serif
    "font.serif": "Linux Libertine O",  # Use "Libertine" as the standard font
    "font.size": 9,
    "axes.titlesize": 9,                # LaTeX default is 10pt font.
    "axes.labelsize": 7,                # LaTeX default is 10pt font.
    "legend.fontsize": 7,               # Make the legend font smaller
    "legend.frameon": True,             # Frame around the legend
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pgf.texsystem": "xelatex",         # Use Xelatex which is TTF font aware
    "pgf.rcfonts": False,               # Use pgf.preamble, ignore standard RC
    "pgf.preamble": [
        r'\usepackage{fontspec}',
        r'\usepackage{unicode-math}',
        r'\usepackage{libertine}',
        r'\setmainfont{Linux Libertine O}',
        r'\setmathfont{Linux Libertine O}',
    ]
}


def sigconf_settings():
    # Set SIGCONF settings.
    matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
    matplotlib.rcParams.update(SIGCONF_RCPARAMS)
    print('SIGCONF settings loaded.')


def load(path):
    print(f'Loading {path}.')
    with open(path, 'rb') as f:
        return pickle.load(f)


def format_time(t):
    match = re.match(r'(\d\d)-(\d\d)', t)
    h, m = match.group(1), match.group(2)
    hour = f'{int(h) - 12}' if int(h) > 12 else h
    minute = f'{int(m)-1:02}'
    return ':'.join([hour, minute])
