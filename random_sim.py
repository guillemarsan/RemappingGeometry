import argparse
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Simulation of one point")
    parser.add_argument("--dimension", type=int, default=6,
                        help="Dimensionality of inputs")
    parser.add_argument("--nb_neurons", type=int, default=64,
                        help="Number of neurons")                   
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed")
    parser.add_argument("--dir", type=str, default='./out/',
                        help="Directory to dump output")
    
    
    args = parser.parse_args()
    np.random.seed(seed=args.seed)

    d = args.dimension
    n = args.nb_neurons

    D = np.random.normal(size=(d,n))

    prefix = "randclosed-"
    name = prefix + "load-polyae-dim-" + str(args.dimension) + "-n-" + str(n) + "-s-" + str(args.seed)
    basepath = args.dir + name
    
    filepath = "%s.npy" % basepath
    np.save(filepath, D)

