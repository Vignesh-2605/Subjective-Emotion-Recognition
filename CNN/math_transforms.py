import numpy as np
from scipy.signal import lti, impulse

def compute_fft(x, sr):
    N = len(x)
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(N, 1/sr)
    return freqs, np.abs(X), np.angle(X), X

def energy_time_domain(x):
    return np.sum(x**2)

def energy_freq_domain(X):
    return np.sum(np.abs(X)**2) / len(X)

def z_transform(x, z=1.01):
    n = np.arange(len(x))
    return np.real(np.sum(x * (z ** (-n))))

def laplace_system_response():
    system = lti([1], [1, 3, 2])
    t, y = impulse(system)
    return np.mean(y), np.max(y)
