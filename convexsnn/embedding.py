import numpy as np
from scipy.stats import ortho_group

def get_embedding(dbbox, dinput, env, input_amp, variance=-1, input_scale = True):

    np.random.seed(env)
    if variance != -1:
        noise = np.random.normal(0, variance, (dbbox,dinput))
        Q = np.eye(dbbox)
        Basis = Q[:,:dinput] + noise
    else:
        Q = ortho_group.rvs(dbbox)
        Basis = Q[:,:dinput]

    # Scale
    Basis = Basis/np.linalg.norm(Basis,axis=0)
    if input_scale:
        Basis = np.sqrt(dbbox)*Basis

    Theta = input_amp*Basis

    return Theta