import argparse
import time
import json
from urllib.request import DataHandler

import matplotlib.pyplot as plt
import numpy as np

from convexsnn.ConvexSNN import ConvexSNN
from convexsnn.input import get_input
from convexsnn.network import get_model
import convexsnn.plot as plot

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Simulation of one point")
    parser.add_argument("--dim_input", type=int, default=3,
                        help="Dimensionality of inputs")
    parser.add_argument("--nb_neurons", type=int, default=6,
                        help="Number of neurons")
    parser.add_argument("--dim_output", type=int, default=3,
                        help="Dimensionality of outputs")
    parser.add_argument("--input", type=str, default='dis-circle',
                        help="Type of input")
    parser.add_argument("--model", type=str, default='cone-polyae',
                        help="Type of model")
    parser.add_argument("--input_amp", type=float, default=1,
                        help="Amplitude of input")
    parser.add_argument("--noise_amp", type=float, default=1,
                        help="Amplitude of noise")
    parser.add_argument("--seed", type=int, default=666,
                        help="Random seed")
    parser.add_argument("--dir", type=str, default='./out/',
                        help="Directory to dump output")
    parser.add_argument("--plot", action='store_true', default=True,
                        help="Plot the results")
    parser.add_argument("--gif", action='store_true', default=True,
                        help="Generate a gif of the bbox")
    
    args = parser.parse_args()
    np.random.seed(seed=args.seed)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    name = args.model + "-" + args.input + "-" + str(args.input_amp)
    basepath = args.dir + timestr + "-" + name
    results = dict(datetime=timestr, basepath=basepath, args=vars(args))

    inp = args.dim_input
    n = args.nb_neurons
    out = args.dim_output

    model, D, G = get_model(inp,n,out,connectivity=args.model)

    # Construction of the input
    x, dx, t, dt, time_steps = get_input(inp, type=args.input, amp=args.input_amp)
    c = model.lamb*x + dx

    # Construction of the noise
    b = np.random.rand(out)*args.noise_amp
    I = G @ b

    # Simulate/train the model
    x0 = x[:,0]
    y0 = -b/model.lamb
    V0 = model.F @ x0 - G @ y0

    print('Simulating model...')
    V, s, r = model.simulate(c, I, V0=V0, dt=dt, time_steps=time_steps)

    # Decode y
    y = D @ r - np.expand_dims(b / model.lamb, axis=-1)

    # Save results 
    results['y_end'] = y[:,-1].tolist()
    results['error'] = np.linalg.norm(V - (model.F @ x - G @ y))
    
    filepath = "%s.json" % basepath
    with open(filepath, "w") as file_handle:
        json.dump(results, file_handle, indent=4)

    # Plot
    if args.plot:
        plot.plot_neuroscience(x, y, V, s, t, basepath)

        if inp == 1 and out == 1:
            plot.plot_iofunc(x, y, model.F, G, model.T, basepath)

        
        if args.input == 'cst' and (out == 2 or out==3):
            print('Generating bounding box plot...')
            if out == 2:
                plot.plot_1dbbox(x[:,-1], y, model.F, G, model.T, basepath, plotx=(args.model == 'randae' or args.model == 'polyae'))
            else:
                plot.plot_2dbboxproj(x[:,-1], y, model.F, G, model.T, basepath, plotx=(args.model == 'randae' or args.model == 'polyae'))

        if inp == 1 or ((args.input == 'circle' or args.input == 'dis-circle') and inp == 2):
            if args.input == 'circle' and inp == 2:
                p = np.arctan2(x[1,:],x[0,:])
                p[p < 0] += 2*np.pi
            else:
                p = x[0,:]

            plot.plot_1drfs(p, r, dt, basepath, pad=100)
            plot.plot_1dspikebins(p, s, 25, basepath, pad=100)
        
        if (args.input == 'spiral' or args.input == 'dis-spiral') and inp == 3:
            p = np.cos(np.arcsin(x[2,:]))*x[:2,:]
            plot.plot_2drfs(p, r, dt, basepath)
            plot.plot_2dspikebins(p, s, 1000, basepath)

    if args.gif and out == 2:
        print('Generating gif...')
        plot.plot_1danimbbox(x, y, model.F, G, model.T, basepath, plotx=(args.model == 'randae' or args.model == 'polyae'))