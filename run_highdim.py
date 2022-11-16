import argparse
import time
import json
import string
import random

import numpy as np
from convexsnn.Codifier import ProjectionCod, TorusCod


from convexsnn.embedding import get_embedding
from convexsnn.network import get_model
from convexsnn.current import get_current
from convexsnn.path import get_path
import convexsnn.plot as plot

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Simulation of one point")
    parser.add_argument("--dim_pcs", type=int, default=2,
                        help="Dimensionality of inputs")
    parser.add_argument("--model", type=str, default='randclosed-load-polyae',
                        help="Type of model")  
    parser.add_argument("--nb_neurons", type=int, default=256,
                        help="Number of neurons")
    parser.add_argument("--dim_bbox", type=int, default=8,
                        help="Dimensionality of outputs") 
    parser.add_argument("--load_id", type=int, default=1,
                        help="In case of load, id of the bbox to load")
    parser.add_argument('--input_dir', nargs='+', type=float, default=[0.],
                        help="Direction of the input")
    parser.add_argument("--input_amp", type=float, default=1.,
                        help="Amplitude of input")
    parser.add_argument("--input_scale", action='store_true', default=True,
                        help="Scale the input by sqrtdbbox")
    parser.add_argument("--cod_scale", type=float, default=0.5,
                        help="Periodicity of the Torus encoding")
    parser.add_argument("--cod_type", type=str, default='square',
                        help="Type of Torus encoding (square, twisted, rhombus, 6D)")
    parser.add_argument('--embed_affine', action='store_true', default=False,
                        help="The embedding happens in an affine way")
    parser.add_argument('--current_neurons', nargs='+',type=float,default=[0],
                        help="Neurons to recieve input current")
    parser.add_argument('--current_amp', type=float, default=0.,
                        help="Amplitude of the current input")
    parser.add_argument("--noise_amp", type=float, default=0.,
                        help="Amplitude of noise")
    parser.add_argument("--decoder_amp", type=float, default=0.2,
                        help="Amplitude of decoder matrix D")
    parser.add_argument("--thresh_amp", type=float, default=1.25,
                        help="Amplitude of the thresholds")    
    parser.add_argument('--thresh_lognorm', action='store_true', default=False,
                        help="The thresholds are taken from a lognormal distribution")                
    parser.add_argument("--seed", type=int, default=666,
                        help="Random seed")
    parser.add_argument("--dir", type=str, default='./out/',
                        help="Directory to dump output")
    parser.add_argument("--plot", action='store_true', default=True,
                        help="Plot the results")
    parser.add_argument("--gif", action='store_true', default=False,
                        help="Generate a gif of the bbox")
    parser.add_argument("--save", action='store_true', default=True,
                        help="Save V, s, r and Th matrices")

    
    args = parser.parse_args()

    timestr = time.strftime("%Y%m%d-%H%M%S")
    code = ''.join(random.choice(string.ascii_letters) for i in range(5))
    name = timestr + "-" + code + "-" + args.model + "-pcs-" + str(args.dim_pcs) + "-bbox-" + str(args.dim_bbox) + "-n-" + str(args.nb_neurons)
    basepath = args.dir + name
    results = dict(datetime=timestr, basepath=basepath, args=vars(args))

    dbbox = args.dim_bbox
    n = args.nb_neurons

    print('Loading model...')
    model, D, G = get_model(dbbox ,n, dbbox,connectivity=args.model, decod_amp=args.decoder_amp, 
                    thresh_amp=args.thresh_amp, load_id=args.load_id, lognormal=args.thresh_lognorm)

    # Construction of the path
    if args.dim_pcs == 1:
        path_type = 'ur'
    else:
        path_type = 'constant'
    p, dp, t, dt, time_steps = get_path(dpcs=args.dim_pcs, type=path_type)   

    # Construction of the input
    print('Codifying input...')
    Codifier = TorusCod()
    x, dx = Codifier.codify(p, dp, scale=args.cod_scale, type=args.cod_type)
    dinput = x.shape[0]

    # Construction of the high dimensional embedding
    print('Embedding input...')
    Theta, k = get_embedding(dbbox, dinput=dinput, input_dir=args.input_dir, input_scale=args.input_scale, input_amp=args.input_amp, D=D,
                        vect='random', affine=args.embed_affine)

    # Embedd
    x = Theta @ x + k
    dx = Theta @ dx
    # x = 0*x + D[:,2,np.newaxis]*(np.sqrt(dbbox)/args.decoder_amp)

    c = model.lamb*x + dx

    # Construction of the current manipulation (noise + experiment)
    np.random.seed(seed=args.seed)
    I, b = get_current(dbbox, t, G, args.noise_amp, args.current_neurons, args.current_amp, vect='neuron', rseed=args.seed)
    #I = I - G@(model.lamb*z + dz)

    # Bias correction for D
    input_amp = np.sqrt(dbbox)*args.input_amp if args.input_scale else args.input_amp
    bias_corr = input_amp/(input_amp+0.5*(args.decoder_amp - args.thresh_amp))

    # Simulate/train the model
    print('Simulating model...')
    x0 = x[:,0]
    r0 = np.linalg.lstsq(bias_corr*D,x0+b[:,0]/model.lamb,rcond=None)[0]
    y0 = D @ r0 - b[:,0] / model.lamb      
    V0 = model.F @ x0 - G @ y0

    V, s, r = model.simulate_one(c, I, V0=V0, r0=r0, dt=dt, time_steps=time_steps)

    # Decode y
    # bias_corr = np.zeros((dbbox,1))
    # bias_corr[0:dinput] = np.sqrt(dinput)/(np.sqrt(dinput)+0.5*(args.decoder_amp - args.thresh_amp))
    # bias_corr[dinput:] = np.sqrt(dbbox-dinput)/(np.sqrt(dbbox-dinput)+0.5*(args.decoder_amp - args.thresh_amp))
    y = bias_corr*(D @ r) - b / model.lamb
    y_disem = (1/(input_amp**2))*Theta.T @ (y - k)
    p_hat = Codifier.decodify(y_disem, scale=args.cod_scale, type=args.cod_type)

    # Save results 
    print('Saving results...')
    results['y_end'] = y[:,-1].tolist()
    results['tracking_error'] = np.mean(np.linalg.norm(y - x, axis=0))
    results['spatial_tracking_error'] = np.mean(np.linalg.norm(p_hat - p, axis=0))

    # PCs
    active_list = np.any(s,axis=1)
    pcs_list = np.where(active_list)[0]
    npcs = pcs_list.shape[0]

    # FRs
    ft = 1
    m = int(ft/dt)
    filter = np.ones(m)
    fr = np.apply_along_axis(lambda m: np.convolve(m, filter, mode='same'), axis=1, arr=s)
    maxfr = np.max(fr[active_list,:], axis=1)
    if np.max(maxfr) <= ft/1e-3:
        print('1 spike/ 1 ms asserted')
    meanfr = np.mean(fr[active_list,:], axis=1)

    results['perpcs'] = npcs/n
    results['pcsidx'] = pcs_list.tolist()
    results['maxfr'] = maxfr.tolist()
    results['meanfr'] = meanfr.tolist()

    results['nb_steps'] = p.shape[1]
    results['dt'] = dt

    if args.save:
        np.savetxt("%s-Th.csv" % basepath, Theta, fmt='%.3e')
        results['Th'] = "%s-Th.csv" % name
        if args.embed_affine:
            np.savetxt("%s-k.csv" % basepath, k, fmt='%.3e')
            results['k'] = "%s-k.csv" % name
        # np.savetxt("%s-V.csv" % basepath, V)
        # results['V'] = "%s-V.csv" % name
        spike_times = np.argwhere(s)
        np.savetxt("%s-stimes.csv" % basepath, spike_times, fmt='%i')
        results['stimes'] = "%s-stimes.csv" % name
        # np.savetxt("%s-r.csv" % basepath, r)
        # results['r'] = "%s-r.csv" % name
    
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

            if n > 49:
                n_vect = np.where(np.any(s,axis=1))[0]
                n_vect = n_vect[:49]
            else: 
                n_vect = np.arange(n)
           
            plot.plot_2drfs(p, r, dt, basepath, n_vect)
            plot.plot_2dspikebins(p, s, dt, 100, basepath, n_vect)
            plot.plot_2drfsth(D, x, p, basepath)

    if args.gif and dbbox == 2:
        print('Generating gif...')
        plot.plot_1danimbbox(x, y, model.F, G, model.T, basepath, plotx=(args.model == 'randae' or args.model == 'polyae'))