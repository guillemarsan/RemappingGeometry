import numpy as np
from scipy.stats import ortho_group

def get_embedding(dbbox, dinput, env, variance=-1, sphere=True):

    np.random.seed(env)
    if sphere:
        if variance != -1:
            Q = np.eye(dbbox)[:,:dinput]
            Basis = np.random.normal(Q, variance, (dbbox,dinput))
            aux = np.linalg.qr(Basis)[0]
            Theta = np.zeros_like(aux)
            for i in np.arange(aux.shape[1]):
                Theta[:,i] = np.sign(aux[0,i])*np.sign(Basis[0,i])*aux[:,i]
        else:
            Basis = ortho_group.rvs(dbbox)
            Theta = Basis[:,:dinput]
    else:
        Theta = np.zeros((dbbox, dbbox))
        for d in np.arange(0, dbbox, 2):
            alpha = np.random.uniform(0, 2*np.pi) if variance == -1 else np.random.normal(0, variance)
            Theta[d:d+2,d:d+2] = np.array([[np.cos(alpha), -np.sin(alpha)],[np.sin(alpha), np.cos(alpha)]])
   
    return Theta