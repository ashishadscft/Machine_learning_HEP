import matplotlib.pyplot as plt
import numpy as np

# Sample data
data_years = ['2016', '2017', '2018']
bins = np.linspace(0, 5, 11)  # Bin edges
data_x = (bins[:-1] + bins[1:]) / 2  # Bin centers
data_y = {year: np.random.poisson(lam=200.0, size=len(data_x)) for year in data_years}  # Simulated data points
data_y_err = {year: np.sqrt(data_y[year]) for year in data_years}  # Poisson errors

# Backgrounds
backgrounds = ['DY', 'Top Quark', 'Diboson', 'Z+jets EW']
colors = ['skyblue', 'lightgreen', 'lightcoral', 'wheat']
bg_data = {bg: np.random.uniform(50, 150, size=(len(data_years), len(data_x))) for bg in backgrounds}

# Create the figure with constrained_layout set to True
fig, axes = plt.subplots(3, 1, figsize=(15, 10), constrained_layout=True)

# Create the plots
for i, year in enumerate(data_years):
    ax = axes[i]

    # Stack the backgrounds
    bg_stack = np.row_stack([bg_data[bg][i] for bg in backgrounds])
    cumulative = np.zeros(len(data_x))
    for j, (bg_vals, color) in enumerate(zip(bg_stack, colors)):
        ax.bar(data_x, bg_vals, bottom=cumulative, color=color, width=bins[1] - bins[0], align='center')
        cumulative += bg_vals

    # Error bars for the data
    ax.errorbar(data_x, data_y[year], yerr=data_y_err[year], fmt='o', color='black', label='Data')

    # Create a twin axis for the ratio plot
    ax_ratio = ax.twinx()

    # Ratio plot
    model = cumulative
    ratio = data_y[year] / model
    ratio_err = data_y_err[year] / model
    ax_ratio.errorbar(data_x, ratio, yerr=ratio_err, fmt='o', color='black')
    ax_ratio.axhline(y=1, color='gray', linewidth=1)

    # Set axis labels and titles
    ax.set_title(f'CMS Preliminary {year}', fontsize=12)
    ax.set_ylabel('Events', labelpad=2)
    ax_ratio.set_ylabel('Data/Model', labelpad=2)
    ax_ratio.set_ylim(0.5, 1.5)  # Adjust the y-limits to match your data

    # Set the ratio plot ticks to the right side
    ax_ratio.yaxis.tick_right()
    ax_ratio.yaxis.set_label_position("right")

    if i == len(data_years) - 1:
        ax.set_xlabel('VBF DNN output')

# Set y-scale to logarithmic for all main axes
for ax in axes:
    ax.set_yscale('log')

# Add the legend only once
handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors] + [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=5)]
labels = backgrounds + ['Data']
axes[0].legend(handles, labels, loc='upper right')

plt.show()
