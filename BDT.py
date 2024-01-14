import numpy as np
import matplotlib.pyplot as plt

# Dummy DNN outputs for signal, backgrounds, and observed data
# Replace these with your actual data
dnn_output_signal = np.random.normal(0.8, 0.1, 1000)
dnn_output_backgrounds = {
    'DY': np.random.normal(0.3, 0.1, 1000),
    'Top Quark': np.random.normal(0.5, 0.1, 1000),
    'Diboson': np.random.normal(0.6, 0.1, 1000),
    'Z+jets': np.random.normal(0.4, 0.1, 1000)
}
dnn_output_data = np.random.normal(0.7, 0.1, 100)

# Define bins for the histograms
bins = np.linspace(0, 1, 21)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Plot stacked backgrounds
plt.figure(figsize=(10, 8))
hist_backgrounds, bin_edges, _ = plt.hist(list(dnn_output_backgrounds.values()),
                                          bins=bins,
                                          stacked=True,
                                          label=list(dnn_output_backgrounds.keys()))

# Plot signal
plt.hist(dnn_output_signal, bins=bins, histtype='step', color='black', label='Signal')

# Histogram the observed data to get counts per bin
data_counts, _ = np.histogram(dnn_output_data, bins)
data_errors = np.sqrt(data_counts)  # Poisson errors

# Overlay the observed data with error bars
plt.errorbar(bin_centers, data_counts, yerr=data_errors, fmt='o', color='red', label='Data')

# Add labels and legend
plt.xlabel('DNN Output')
plt.ylabel('Events')
plt.title('DNN Output Distribution')
plt.legend()

# Show the histogram
plt.show()

# Ratio plot (Data / Background)
# Calculate the sum of the background predictions (the last element in hist_backgrounds)
sum_backgrounds = hist_backgrounds[-1]

# Calculate the ratio of data to the sum of the backgrounds, ensure the output array is float
ratio = np.divide(data_counts, sum_backgrounds, out=np.zeros_like(data_counts, dtype=float), where=sum_backgrounds!=0)
ratio_errors = ratio * np.sqrt((data_errors / data_counts) ** 2 + (np.sqrt(sum_backgrounds) / sum_backgrounds) ** 2)

# Create the ratio plot
plt.figure(figsize=(10, 3))
plt.errorbar(bin_centers, ratio, yerr=ratio_errors, fmt='o', color='black')

# Formatting the ratio plot
plt.xlabel('DNN Output')
plt.ylabel('Data / Background')
plt.title('Ratio of Data to Background')
plt.axhline(y=1, color='gray', linestyle='--')  # Line at ratio=1 for reference
plt.ylim(0, 2)  # Set y-axis limits for clarity

# Show the ratio plot
plt.show()
