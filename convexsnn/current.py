
import numpy as np

def get_current(dbbox, t, G, noise_amp, current_neurons, current_amp):
    # Noise
    ones = np.ones((dbbox,t.shape[0]))
    b = (np.random.rand(dbbox)*noise_amp)[:,None]*ones
    I = G @ b

    # Current manipulation
    start = int(t.shape[0]/5)
    I[current_neurons,start:] += current_amp
    return I, b