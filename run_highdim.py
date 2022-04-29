import argparse
import time
import json
from urllib.request import DataHandler

import matplotlib.pyplot as plt
import numpy as np
from convexsnn.Codifier import ProjectionCod

from convexsnn.network import get_model
from convexsnn.current import get_current
from convexsnn.path import get_path
import convexsnn.plot as plot

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Simulation of one point")
    parser.add_argument("--dim_pcs", type=int, default=2,
                        help="Dimensionality of inputs")
    parser.add_argument("--nb_neurons", type=int, default=16,
                        help="Number of neurons")
    parser.add_argument("--dim_bbox", type=int, default=8,
                        help="Dimensionality of outputs")
    parser.add_argument("--model", type=str, default='load-polyae-proj',
                        help="Type of model")
    parser.add_argument("--load_id", type=int, default=2,
                        help="In case of load, id of the bbox to load")
    parser.add_argument("--input_amp", type=float, default=1.,
                        help="Amplitude of input")
    parser.add_argument('--input_dir', nargs='+', type=float, default=[2],
                        help="Direction of the input")
    parser.add_argument('--current_neurons', nargs='+',type=int,default=[0],
                        help="Neurons to recieve input current")
    parser.add_argument('--current_amp', type=float, default=-1.,
                        help="Amplitude of the current input")
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
    parser.add_argument("--gif", action='store_true', default=False,
                        help="Generate a gif of the bbox")
    
    args = parser.parse_args()
    np.random.seed(seed=args.seed)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    name = args.model + "-pcs-" + str(args.dim_pcs) + "-bbox-" + str(args.dim_bbox) + "-n-" + str(args.nb_neurons)
    basepath = args.dir + timestr + "-" + name
    results = dict(datetime=timestr, basepath=basepath, args=vars(args))

    dbbox = args.dim_bbox
    if args.model == 'grid-polyae': 
        sqrtn = int(np.sqrt(args.nb_neurons))
        n = sqrtn**2
    else:
        n = args.nb_neurons

    model, D, G = get_model(dbbox ,n, dbbox,connectivity=args.model, decod_amp=args.decoder_amp, thresh_amp=args.thresh_amp, load_id=args.load_id)

    # Construction of the path
    if args.dim_pcs == 1:
        path_type = 'ur'
    else:
        path_type = 'uspiral'
    p, dp, t, dt, time_steps = get_path(dpcs=args.dim_pcs, type=path_type)

    # Construction of the input
    if args.dim_pcs == 1 and len(args.input_dir) != dbbox-1:
        Theta = D[:-1,int(args.input_dir[0])][:,None]
    elif args.dim_pcs == 2 and len(args.input_dir) != 2*(dbbox-1):
        selneur = D[:-1,int(args.input_dir[0])][:,None]
        dir2 = np.zeros_like(selneur)
        dir2[-2] = 1
        dir2[-1] = -selneur[-2]/selneur[-1]
        Theta = np.hstack((selneur, dir2))
    else:
        Theta = args.input_dir.reshape((-1,args.dim_pcs))
    Theta = Theta/np.linalg.norm(Theta,axis=0)

    Codifier = ProjectionCod()
    x, dx = Codifier.codify(p, dp, A=args.input_amp, Theta=Theta)
    c = model.lamb*x + dx

    # Construction of the current manipulation (noise + experiment)
    I, b = get_current(dbbox, t, G, args.noise_amp, args.current_neurons, args.current_amp)

    # Simulate/train the model
    x0 = x[:,0]
    r0 = np.linalg.lstsq(D,x0+b[:,0]/model.lamb,rcond=None)[0]
    y0 = D @ r0 - b[:,0] / model.lamb        
    V0 = model.F @ x0 - G @ y0

    print('Simulating model...')
    V, s, r = model.simulate(c, I, V0=V0, r0=r0, dt=dt, time_steps=time_steps)

    # Decode y
    decod = D @ r
    bias_corr = args.input_amp/(args.input_amp+0.5*(args.decoder_amp**2 - 1))
    y = bias_corr*decod - b / model.lamb

    # Save results 
    results['y_end'] = y[:,-1].tolist()
    results['tracking_error'] = np.linalg.norm(y - x)
    
    tf = 0.2
    cutoff = 0
    m = int(tf/dt)
    filter = np.ones(m)*1/m
    pcs = 0
    for i in range(r.shape[0]):
        rf = np.convolve(r[i,:],filter, 'same')
        if np.max(rf) > cutoff:
            pcs += 1
    results['pcs_percentage'] = pcs/n
    
    filepath = "%s.json" % basepath
    with open(filepath, "w") as file_handle:
        json.dump(results, file_handle, indent=4)

    # Plot
    if args.plot:
        print('Generating neuroscience plot...')
        plot.plot_neuroscience(x, y, V, s, t, basepath)
      
        if dbbox == 2 or dbbox ==3:
            print('Generating bounding box plot...')
            if dbbox == 2:
                plot.plot_1dbbox(x[:,-1], y[:,-1:], model.F, G, model.T, basepath, plotx=(args.model == 'randae' or args.model == 'polyae'))
            else:
                plot.plot_2dbboxproj(model.F, G, model.T, args.input_amp, basepath)

        if args.dim_pcs == 1:
            print('Generating 1drfs plot...')

            plot.plot_1drfs(p, r, dt, basepath, pad=0)
            plot.plot_1dspikebins(p, s, 25, basepath, pad=0)
            plot.plot_1drfsth(D, x, p, basepath, pad=0)
        
        if args.dim_pcs == 2:
            print('Generating 2drfs plot...')
           
            plot.plot_2drfs(p, r, dt, basepath)
            plot.plot_2dspikebins(p, s, 100, basepath)
            plot.plot_2drfsth(D, x, p, basepath)

    if args.gif and dbbox == 2:
        print('Generating gif...')
        plot.plot_1danimbbox(x, y, model.F, G, model.T, basepath, plotx=(args.model == 'randae' or args.model == 'polyae'))