import argparse
import numpy as np
import matplotlib.pyplot as plt

def simulate_lloyd(D0,prec=1e-2, crust_sampling=False):
    dim = D0.shape[0]
    n = D0.shape[1]

    print('Dropping sampling...')
    if args.crust_sampling:
        crusts = 10
        r = np.linspace(0,1,crusts,endpoint=False)
        pts = np.random.normal(size=(dim,10000))
        pts = pts/np.linalg.norm(pts,axis=0)
        npts = int(10000/crusts)
        for c in np.arange(crusts-1):
            pts[:,npts*c:npts*(c+1)] = pts[:,npts*c:npts*(c+1)]*r[c+1]
            plt.scatter(pts[0,:], pts[1,:])
        
    else:
        pts = np.random.normal(size=(dim+2,10000))
        norm = np.linalg.norm(pts,axis=0)
        pts = pts/norm
        pts = pts[0:dim] 
    

    print('Lloyds algorithm...')
    Dnew = D0
    Dold = Dnew+1
    while np.linalg.norm(Dnew - Dold) > prec:
        Dold = np.copy(Dnew)
        diffvect = Dold.T - pts.T[:,None]
        weights = np.linalg.norm(diffvect, axis=-1)
        belong = np.argmin(weights,axis=1)
        for i in np.arange(n):
            contr = pts[:,belong==i]
            if contr.shape[1] != 0:
                Dnew[:,i] = np.sum(contr,axis=1)/contr.shape[1]
        plt.scatter(Dnew[0,:],Dnew[1,:])
    return Dnew

def normalize(D):

    norms = np.linalg.norm(D,axis=0)
    normlizer = np.ones_like(norms)
    normlizer[norms>1] = norms[norms>1]
    D = D/normlizer
    return D


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Simulation of one point")
    parser.add_argument("--dimension", type=int, default=3,
                        help="Dimensionality of inputs")
    parser.add_argument("--nb_neurons", type=int, default=150,
                        help="Number of neurons")                   
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--dir", type=str, default='./out/',
                        help="Directory to dump output")
    parser.add_argument("--prec", type=float, default=1e-3,
                        help="Stopping precision")
    parser.add_argument("--crust_sampling", action='store_true', default=True,
                        help="Sample points in crusts")

    
    
    args = parser.parse_args()
    np.random.seed(seed=args.seed)

    prefix = "crust-" if args.crust_sampling else ""
    name = prefix + "load-polyae-proj-dim-" + str(args.dimension) + "-n-" + str(args.nb_neurons) + "-s-" + str(args.seed)
    basepath = args.dir + name

    d = args.dimension
    n = args.nb_neurons

    
    D0 = np.random.normal(size=(d-1,n))
    D0 = normalize(D0)
        

    print('Simulating model...')
    D = simulate_lloyd(D0, prec=args.prec, crust_sampling=args.crust_sampling)

    vect = np.sqrt(1-np.linalg.norm(D,axis=0)**2)
    D = np.concatenate((D,vect[None,:]),axis=0)
    

    filepath = "%s.npy" % basepath
    np.save(filepath, D)

