import argparse
import numpy as np

def simulate_repulsion(D0,prec=1e-2):
    dt = 1e-2
    Dnew = D0
    Dold = Dnew+1
    while np.linalg.norm(Dnew - Dold) > prec:
        Dold = Dnew
        diffvect = Dold.T - Dold.T[:,None]
        weights = np.linalg.norm(diffvect, axis=-1)
        weights[weights != 0] = weights[weights != 0]**(-2)
        forces = np.sum(diffvect*weights[:,:,None],axis=1).T
        Dnew = Dold - dt*forces
        Dnew = normalize(Dnew)
    return Dnew

def normalize(D):
    norms = np.linalg.norm(D,axis=0)
    normlizer = np.ones_like(norms)
    normlizer[norms>1] = norms[norms>1]
    # D = D/normlizer
    
    negative = D[-1,:] < 0
    D[-1,negative] = 0
    D = D/norms
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

    name = "load-polyae-dim-" + str(args.dimension) + "-n-" + str(args.nb_neurons)
    basepath = args.dir + name

    d = args.dimension
    n = args.nb_neurons

    D0 = np.random.normal(size=(d,n))
    D0 = normalize(D0)


    print('Simulating model...')
    D = simulate_repulsion(D0, prec=args.prec)

    #D = np.concatenate((D,np.ones((1,D.shape[1]))*0.5),axis=0)
    #D = D/np.linalg.norm(D,axis=0)
    

    filepath = "%s.npy" % basepath
    np.save(filepath, D)

