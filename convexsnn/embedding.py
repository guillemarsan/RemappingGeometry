
import numpy as np
from scipy.stats import ortho_group

def get_embedding(dbbox, dinput, input_dir, input_amp, D, vect='neuron', affine=False):

    Basis = np.zeros((dbbox, dinput))

    if not affine:
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
        k = np.zeros((dbbox,1))
    else:
        Q = np.eye(dbbox)
        Basis = Q[:,:dinput]

        np.random.seed(seed=int(input_dir[0]))
        affine_dir = np.random.rand(dbbox)
        affine_dir[:dinput] = 0
        affine_dir = affine_dir/np.linalg.norm(affine_dir)
        k_mag = np.random.uniform(-5,5)
        k = k_mag*affine_dir[:,np.newaxis]
        k = np.zeros((dbbox,1))

    Theta = input_amp*Basis

    return Theta, k