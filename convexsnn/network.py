
import numpy as np
from convexsnn.ConvexSNN import ConvexSNN

def get_model(inp, n, out, connectivity):

    ones = np.ones((out,n))
    D = ones

    # Construction of the model
    if connectivity == 'custom':
        F = np.random.rand(n,inp)
        F[1,:] *= -1
        nF = np.linalg.norm(F, axis=0)
        F = F/nF

        D = np.random.normal(size=(out,n))
        D[0,:] *= -1
          
    elif connectivity == 'randae':
        D = np.random.normal(size=(out,n))
    
    elif connectivity == 'bowl-randae':
        D[0,:] = np.random.normal(size=(1,n))
        if inp > 1: D[1,:] = np.random.normal(size=(1,n))

    elif connectivity == 'cone-polyae':
        deg = np.arange(n) * 2*np.pi / n + 1e-3
        D[0,:] = np.sin(deg)*ones[0,:]
        if inp > 1: D[1,:] = np.cos(deg)*ones[1,:]

    elif connectivity == 'bowl-polyae':
        deg = np.arange(n) * 2*np.pi / n + 1e-3
        highdeg = np.arange(n) * np.pi /(2*n) + 1e-3
        nint = (np.max(highdeg)-highdeg)
        D[0,:] = nint*np.sin(deg)*ones[0,:]
        if inp > 1: D[1,:] = nint*np.cos(deg)*ones[1,:]
        if inp > 2:
            D[2,:] = np.sin(highdeg)*ones[2,:]
    
    lamb = 100
    nD = np.linalg.norm(D, axis=0)
    D = D/nD
    if connectivity != 'custom': 
        F = D.transpose()
    G = D.transpose()
    T = np.ones(n) * 1/2*(np.linalg.norm(D, axis=0)**2)

    Om = -G @ D 

    model = ConvexSNN(lamb, F, Om, T)
    return model, D, G