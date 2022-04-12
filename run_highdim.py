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
    parser.add_argument("--dim_pcs", type=int, default=1,
                        help="Dimensionality of inputs")
    parser.add_argument("--nb_neurons", type=int, default=16,
                        help="Number of neurons")
    parser.add_argument("--dim_bbox", type=int, default=4,
                        help="Dimensionality of outputs")
    parser.add_argument("--model", type=str, default='bowl-randae',
                        help="Type of model")
    parser.add_argument("--input_amp", type=float, default=1.,
                        help="Amplitude of input")
    parser.add_argument('--input_dir', nargs='+', type=float, default=[1,0],
                        help="Direction of the input")
    parser.add_argument("--noise_amp", type=float, default=1.,
                        help="Amplitude of noise")
    parser.add_argument("--decoder_amp", type=float, default=1,
                        help="Amplitude of decoder matrix D")
    parser.add_argument("--thresh_amp", type=float, default=1,
                        help="Amplitude of the thresholds")                    
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
    name = args.model + "-pcs-" + str(args.dim_pcs) + "-bbox-" + str(args.dim_bbox)
    basepath = args.dir + timestr + "-" + name
    results = dict(datetime=timestr, basepath=basepath, args=vars(args))

    dbbox = args.dim_bbox
    if args.model == 'grid-polyae': 
        sqrtn = int(np.sqrt(args.nb_neurons))
        n = sqrtn**2
    else:
        n = args.nb_neurons

    model, D, G = get_model(dbbox ,n, dbbox,connectivity=args.model, decod_amp=args.decoder_amp, thresh_amp=args.thresh_amp)

    # Construction of the input
    if args.dim_pcs == 1:
        input_type = 'semicircle'
    x, dx, t, dt, time_steps = get_input(dbbox, type=input_type, amp=args.input_amp, dir=args.input_dir)
    c = model.lamb*x + dx

    # Construction of the noise
    b = np.random.rand(dbbox)*args.noise_amp
    I = G @ b

    # Simulate/train the model
    x0 = x[:,0]
    r0 = np.linalg.lstsq(D,x0+b/model.lamb,rcond=None)[0]
    y0 = D @ r0 - b / model.lamb        
    V0 = model.F @ x0 - G @ y0

    print('Simulating model...')
    V, s, r = model.simulate(c, I, V0=V0, r0=r0, dt=dt, time_steps=time_steps)

    # Decode y
    y = D @ r - b[:,None] / model.lamb

    # Save results 
    results['y_end'] = y[:,-1].tolist()
    results['error'] = np.linalg.norm(V - (model.F @ x - G @ y))
    
    filepath = "%s.json" % basepath
    with open(filepath, "w") as file_handle:
        json.dump(results, file_handle, indent=4)

    # Plot
    if args.plot:
        plot.plot_neuroscience(x, y, V, s, t, basepath)
      
        if dbbox == 2 or dbbox ==3:
            print('Generating bounding box plot...')
            if dbbox == 2:
                plot.plot_1dbbox(x[:,-1], y[:,-1:], model.F, G, model.T, basepath, plotx=(args.model == 'randae' or args.model == 'polyae'))
            else:
                plot.plot_2dbboxproj(x[:,-1], y[:,-1:], model.F, G, model.T, basepath, plotx=(args.model == 'randae' or args.model == 'polyae'))

        if args.dim_pcs == 1:
          
            p = np.arctan2(x[-1,:],x[0,:])

            plot.plot_1drfs(p, r, dt, basepath, pad=100)
            plot.plot_1dspikebins(p, s, 25, basepath, pad=100)
        
        if args.dim_pcs == 2:
            # TODO
            p = np.cos(np.arcsin(x[2,:]))*x[:2,:]
            plot.plot_2drfs(p, r, dt, basepath)
            plot.plot_2dspikebins(p, s, 100, basepath)

    if args.gif and dbbox == 2:
        print('Generating gif...')
        plot.plot_1danimbbox(x, y, model.F, G, model.T, basepath, plotx=(args.model == 'randae' or args.model == 'polyae'))