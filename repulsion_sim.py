import argparse
import numpy as np
import matplotlib.pyplot as plt

def simulate_repulsion(D0,prec=1e-2, closed=False):
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
        Dnew = normalize(Dnew,closed)
    return Dnew

def normalize(D, closed):
    norms = np.linalg.norm(D,axis=0)
    if not closed:
        negative = D[-1,:] < 0
        D[-1,negative] = 0
    D = D/norms
    return D


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Simulation of one point")
    parser.add_argument("--dimension", type=int, default=4,
                        help="Dimensionality of inputs")
    parser.add_argument("--nb_neurons", type=int, default=8,
                        help="Number of neurons")                   
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed")
    parser.add_argument("--dir", type=str, default='./out/',
                        help="Directory to dump output")
    parser.add_argument("--prec", type=float, default=1e-3,
                        help="Stopping precision")
    parser.add_argument("--double", action='store_true', default=False,
                        help="Cut in half the result")
    parser.add_argument("--closed", action='store_true', default=True,
                        help="Do it in a complete sphere")

    
    
    args = parser.parse_args()
    np.random.seed(seed=args.seed)


    d = args.dimension
    n = args.nb_neurons

    good_cut = False

    while not good_cut:
        if args.double:
            D0 = np.random.normal(size=(d,2*n))
        else: D0 = np.random.normal(size=(d,n))
        D0 = normalize(D0, args.closed)


        print('Simulating model...')
        D = simulate_repulsion(D0, prec=args.prec, closed=args.closed)
        plt.scatter(D[0,:], D[1,:])

        if args.double:
            D = D[:,D[-1,:] >= 0]
            good_cut = D.shape[1] == n
        else: good_cut = True

    prefix = "doub-" if args.double else ("closed-" if args.closed else "")
    name = prefix + "load-polyae-dim-" + str(args.dimension) + "-n-" + str(n) + "-s-" + str(args.seed)
    basepath = args.dir + name
    
    filepath = "%s.npy" % basepath
    np.save(filepath, D)

