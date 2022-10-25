
import numpy as np
from scipy.stats import ortho_group

def get_basis(dbbox, dinput, input_dir, input_amp, D, vect='neuron', normalize=False):

    Basis = np.zeros((dbbox, dinput))

    # In case of a neuron specified
    if vect == 'neuron':
        selneur = D[:-1,int(input_dir[0])]
        Basis[:-1,0] = selneur
        if dinput > 2:
            dir2 = np.zeros_like(selneur)
            dir2[-2] = 1
            dir2[-1] = -selneur[-2]/selneur[-1]
            Basis[:-1,1] = dir2
        if dinput > 3:
            dir3 = np.zeros_like(selneur)
            dir3[-3] = -(selneur[-2]**2+selneur[-1]**2)/(selneur[-2]*selneur[-3])
            dir3[-2] = 1
            dir3[-1] = selneur[-1]/selneur[-2]
            Basis[:-1,2] = dir3
        Basis[-1,-1] = 1
    elif vect == 'random':
        np.random.seed(seed=int(input_dir[0]))
        Q = ortho_group.rvs(dbbox)
        Basis = Q[:,:dinput]
    elif vect == 'random_variance':
        np.random.seed(seed=int(input_dir[0]))
        noise = np.random.normal(0, 0.25, (dbbox,dinput))
        Q = np.eye(dbbox)
        Basis = Q[:,:dinput] + noise
    else:
        Basis = np.array(input_dir).reshape((dbbox, dinput))

    # Scale
    Basis = Basis/np.linalg.norm(Basis,axis=0)
    Theta = input_amp*Basis
    if normalize:
        Theta = np.sqrt(D.shape[1])*Theta

    return Theta