import argparse
import numpy as np
import matplotlib.pyplot as plt

def simulate_lloyd(D0,prec=1e-2):
    dim = D0.shape[0]
    n = D0.shape[1]

    pts = np.random.normal(size=(dim,10000))
    pts = pts[:,np.linalg.norm(pts,axis=0) < 1]
    # plt.scatter(pts[0,:],pts[1,:])

    Dnew = D0
    Dold = Dnew+1
    while np.linalg.norm(Dnew - Dold) > prec:
        Dold = np.copy(Dnew)
        diffvect = Dold.T - pts.T[:,None]
        weights = np.linalg.norm(diffvect, axis=-1)
        belong = np.argmin(weights,axis=1)
        for i in np.arange(n):
            contr = pts[:,belong==i]
            Dnew[:,i] = np.sum(contr,axis=1)/contr.shape[1]
    #    plt.scatter(Dnew[0,:],Dnew[1,:])
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
    parser.add_argument("--nb_neurons", type=int, default=11,
                        help="Number of neurons")                   
    parser.add_argument("--seed", type=int, default=666,
                        help="Random seed")
    parser.add_argument("--dir", type=str, default='./out/',
                        help="Directory to dump output")
    parser.add_argument("--prec", type=float, default=1e-3,
                        help="Stopping precision")

    
    
    args = parser.parse_args()
    np.random.seed(seed=args.seed)

    name = "load-polyae-proj-dim-" + str(args.dimension) + "-n-" + str(args.nb_neurons)
    basepath = args.dir + name

    d = args.dimension
    n = args.nb_neurons

    D0 = np.random.normal(size=(d-1,n))
    D0 = normalize(D0)

    print('Simulating model...')
    D = simulate_lloyd(D0, prec=args.prec)

    vect = np.sqrt(1-np.linalg.norm(D,axis=0)**2)
    D = np.concatenate((D,vect[None,:]),axis=0)
    

    filepath = "%s.npy" % basepath
    np.save(filepath, D)

