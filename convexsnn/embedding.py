import numpy as np
# from scipy.stats import ortho_group

def get_embedding(dbbox, dinput, env, ortho=True, variance=-1):

    np.random.seed(env)
    if variance != -1:
        Q = np.eye(dbbox)[:,:dinput]
        Basis = np.random.normal(Q, variance, (dbbox,dinput))
    else:
        Basis = np.random.uniform(size=(dbbox,dinput))
        
    aux = np.linalg.qr(Basis)[0]
    Theta = np.zeros_like(aux)
    for i in np.arange(aux.shape[1]):
        Theta[:,i] = np.sign(aux[0,i])*np.sign(Basis[0,i])*aux[:,i]
   
    return Theta