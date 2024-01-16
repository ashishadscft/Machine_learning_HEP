import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
from scipy.stats import poisson, norm
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
data_VBFSR = np.random.normal(0, 1, (1000, 10))  # Data for VBFSR
data_VBFSB = np.random.normal(0, 1, (800, 10))   # Data for VBF-SB

# Create a DNN model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(np.concatenate((data_VBFSR, data_VBFSB)), np.concatenate((np.ones(1000), np.zeros(800))), epochs=10, batch_size=32, verbose=0)

# Apply the DNN model to get scores
scores_VBFSR = model.predict(data_VBFSR).flatten()
scores_VBFSB = model.predict(data_VBFSB).flatten()

# Define bins
bins = np.linspace(0, 1, 11)

# Bin the scores
binned_scores_VBFSR = np.digitize(scores_VBFSR, bins)
binned_scores_VBFSB = np.digitize(scores_VBFSB, bins)

# Function to calculate the combined likelihood for signal and background
def combined_likelihood(params, data, bins):
    mu_signal, sigma_signal, mu_background, sigma_background = params
    prob_signal = norm.pdf(data, mu_signal, sigma_signal)
    prob_background = norm.pdf(data, mu_background, sigma_background)
    total_prob = prob_signal + prob_background
    binned_prob, _ = np.histogram(total_prob, bins=bins)
    likelihood = -np.sum(poisson.logpmf(binned_prob, np.mean(binned_prob)))
    return likelihood

# Perform a binned maximum likelihood fit
initial_params = [0, 1, 0, 1]
result = minimize(combined_likelihood, initial_params, args=(np.concatenate((scores_VBFSR, scores_VBFSB)), bins))

# Plotting
plt.figure(figsize=(15, 5))

# Plot 1: Distribution of DNN Scores
plt.subplot(1, 3, 1)
plt.hist(scores_VBFSR, bins=10, alpha=0.5, label='VBFSR')
plt.hist(scores_VBFSB, bins=10, alpha=0.5, label='VBF-SB')
plt.title('Distribution of DNN Scores')
plt.xlabel('DNN Score')
plt.ylabel('Frequency')
plt.legend()

# Plot 2: Histogram of Binned Scores
plt.subplot(1, 3, 2)
plt.hist(binned_scores_VBFSR, bins=len(bins)-1, alpha=0.5, label='VBFSR')
plt.hist(binned_scores_VBFSB, bins=len(bins)-1, alpha=0.5, label='VBF-SB')
plt.title('Histogram of Binned Scores')
plt.xlabel('Bin')
plt.ylabel('Frequency')
plt.legend()

# Plot 3: Likelihood Fit Results (Assuming Normal Distribution for Visualization)
plt.subplot(1, 3, 3)
x = np.linspace(0, 1, 100)
plt.plot(x, norm.pdf(x, *result.x[:2]), label='Signal')
plt.plot(x, norm.pdf(x, *result.x[2:]), label='Background')
plt.title('Likelihood Fit Results')
plt.xlabel('DNN Score')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()
plt.show()

# Print results
print("Fit results:", result.x)
