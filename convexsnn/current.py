
import numpy as np

def get_current(dbbox, t, G, noise_amp, current_neurons, current_amp, vect='neuron', rseed=0):
    # Noise
    ones = np.ones((dbbox,t.shape[0]))
    np.random.seed(seed=rseed)
    # b = (np.random.rand(dbbox)*noise_amp)[:,None]*ones # Constant noise
    b = (np.random.rand(dbbox,t.shape[0])-0.5)*noise_amp
    I = G @ b

    # Current manipulation
    start = 0 #int(t.shape[0]/5)
    if vect == 'neuron':
        idx = current_neurons
    elif vect == 'fixed_per':
        nneurons = int(current_neurons[0]*I.shape[0])
        idx = np.arange(nneurons)
    elif vect == 'rand_per':
        np.random.seed(seed=rseed)
        nneurons = int(current_neurons[0]*I.shape[0])
        idx = np.random.choice(I.shape[0], size=nneurons, replace=False)

    I[idx,start:] += current_amp
    return I, b