
import numpy as np

def get_basis(dbbox, dinput, input_dir, input_amp, D, vect=False):

    Basis = np.zeros((dbbox, dinput))

    # In case of a neuron specified
    if not vect:
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
            
    else:
        Basis = np.array(input_dir).reshape((dbbox, dinput))

    # Scale
    Basis = Basis/np.linalg.norm(Basis,axis=0)
    Theta = input_amp*Basis

    return Theta