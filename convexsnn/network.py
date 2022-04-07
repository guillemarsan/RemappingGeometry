
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

    elif connectivity == 'grid-polyae' or connectivity == 'gridproj-polyae':
        ptr = int(np.sqrt(n))
        side = ptr+1
        step = 1
        rows = np.arange(-side/2+step, side/2, step=step)
        ppts = np.dstack(np.meshgrid(rows, -rows)).reshape(-1,2).T
        D[0,:] = ppts[0,:]
        if inp > 1: D[1,:] = ppts[1,:]
        if inp > 2: 
            r = np.sqrt(1/2)*side
            if connectivity == 'gridproj-polyae':
                D[2,:] = np.sqrt(r**2 - np.linalg.norm(ppts, axis=0)**2)
            else:
                D[2,:] = (r/2)*ones[2,:]

    elif connectivity == 'cirgrid-polyae' or connectivity == 'cirgridproj-polyae':
        ptr = int(np.sqrt(n))
        side = ptr+1
        lvls = int(ptr/2)
        step = side/(2*(lvls+1))
        radi = np.arange(lvls)[::-1]+1*step

        # r = np.sqrt(1/2)*side
        # lvls = int(ptr/2)
        # step = np.pi/(2*(lvls+1))
        # radi = r*np.cos((np.arange(lvls)+1)*step)
        
        id = 0
        idnew = 0
        for l in np.arange(lvls):
            neurlvl = int(4*(ptr-2*l) - 4)
            deg = np.arange(neurlvl) * 2*np.pi / neurlvl + (3/4*np.pi) + 1e-3
            idnew += neurlvl
            D[0,id:idnew] = radi[l]*np.sin(deg)*ones[0,id:idnew]
            if inp > 1: D[1,id:idnew] = radi[l]*np.cos(deg)*ones[1,id:idnew]
            id = idnew
        if ptr % 2 == 1:
            D[:2,-1] = ones[:2,-1]*0

        if inp > 2: 
            r = np.sqrt(1/2)*side
            if connectivity == 'cirgridproj-polyae':
                D[2,:] = np.sqrt(r**2 - np.linalg.norm(D, axis=0)**2)
            else:
                D[2,:] = (r/2)*ones[2,:]



    
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