import matplotlib.pyplot as plt

from math import sqrt


def plot(results, metric, title, models, labels, lines, colors, position,
         regions='regions', ax=None, fig=None, ylim=None, x_logscale=False,
         ylabel=True, xlabel=True, legend=True):
    # Extract data from results.
    x = results['x']
    all_models = results['models']
    values = results['results'][metric]
    if 'displacement' in metric:
        values = - (values * 5 / 2 - 5 / 2)
    mean = values.mean(axis=(1, 2))
    std = values.std(axis=(1, 2))
    std = std / sqrt(values.shape[1] * values.shape[2])  # Standard error.

    # Plot each model results.
    for model, label, line, color in zip(models, labels, lines, colors):
        i = all_models.index(model)
        print(model, i)
        y, yerr = mean[i], std[i]
        # Draw lines.
        ax.plot(x, y,
                label=label,
                c=color,
                lw=2,
                linestyle=line)
        ax.fill_between(x, y1=y+yerr, y2=y-yerr, alpha=0.2, color=color)

    # Set log scales.
    if x_logscale:
        ax.set_xscale('log')
    # if y_logscale:
    #     ax.set_yscale('log')

    # Axes settings.
    if position == 'bottom':
        s = f'Number of observed {regions}'
        ax.set_xlabel(s + r' $|\mathcal{O}|$')
    elif position == 'top':
        if legend:
            ax.legend()
        # Hide x-tick labels.
        ax.xaxis.set_major_formatter(plt.NullFormatter())
        # Set title.
        ax.set_title(title)
    # Set y-axis label.
    if ylabel:
        if 'mae' in metric:
            ax.set_ylabel(r'$\mathrm{MAE}$ [\%]')
        elif 'displacement' in metric:
            ax.set_ylabel('Average Displacement')
        else:
            ax.set_ylabel(r'Accuracy [\%]')
    # Set limits and ticks of y-axis.
    if ylim is not None:
        ax.set_ylim(ylim)
    ticks = ax.get_yticks()
    if 'displacement' not in metric:
        ax.set_yticklabels([f'{t*100:.0f}' for t in ticks])
    else:
        ax.set_yticklabels([f'{t:.1f}' for t in ticks])
    # ax.minorticks_off()  # Remove minor ticks.
    ax.grid()
    # ax.xaxis.grid(True, which='both')
    # ax.yaxis.grid(True, which='major')
