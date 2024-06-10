import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve

# Generate synthetic data: sine wave + noise
np.random.seed(0)
time = np.linspace(0, 1, 500)
data = np.sin(2 * np.pi * 10 * time) + np.random.normal(0, 0.5, 500)

# Moving Average Smoothing
def moving_average(data, window_size):
    window = np.ones(window_size) / window_size
    return convolve(data, window, mode='same')

# Exponential Smoothing
def exponential_smoothing(data, alpha):
    smoothed_data = np.zeros_like(data)
    smoothed_data[0] = data[0]
    for t in range(1, len(data)):
        smoothed_data[t] = alpha * data[t] + (1 - alpha) * smoothed_data[t-1]
    return smoothed_data

# Gaussian Smoothing
def gaussian_smoothing(data, sigma):
    return gaussian_filter(data, sigma=sigma)

# Apply smoothing techniques
ma_data = moving_average(data, window_size=10)
print('data', data, 'len', len(data))
print('ma data', ma_data, 'len', len(ma_data))
es_data = exponential_smoothing(data, alpha=0.1)
gs_data = gaussian_smoothing(data, sigma=10)

# Plot results
plt.figure(figsize=(14, 7))
plt.plot(time, data, label='Original Data', alpha=0.5)
plt.plot(time, ma_data, label='Moving Average', alpha=0.8)
plt.plot(time, es_data, label='Exponential Smoothing', alpha=0.8)
plt.plot(time, gs_data, label='Gaussian Smoothing', alpha=0.8)
plt.legend()
plt.title('Comparison of Smoothing Techniques')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.savefig('test.png')
