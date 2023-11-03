
import numpy as np

def get_noise(dbbox, t, G, noise_amp, noise_seed=0):
    # Noise
    np.random.seed(seed=noise_seed)
    b = (np.random.rand(dbbox,t.shape[0])-0.5)*noise_amp
    I = G @ b
    return I,b

def get_current_random(n, t, current_per, current_amp, current_seed=0):

    # Current manipulation
    start = 0 #int(t.shape[0]/5)
    
    nneurons = int(current_per*n)
    np.random.seed(seed=current_seed)
    idx = np.random.choice(n, size=nneurons, replace=False)

    current = np.zeros((n,t.shape[0]))
    current[idx,start:] = current_amp
    return current

def get_current(n, t, tagged_idx, current_amp):

    # Current manipulation
    start = 0 
    
    current = np.zeros((n,t.shape[0]))
    current[tagged_idx,start:] = current_amp
    return current