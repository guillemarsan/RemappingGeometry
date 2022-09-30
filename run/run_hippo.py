import argparse
import time
import json
import string
import random

import numpy as np
from convexsnn.Codifier import ProjectionCod, TorusCod

from convexsnn.basis import get_basis
from convexsnn.network import get_model
from convexsnn.current import get_current
from convexsnn.path import get_path
import convexsnn.plot as plot

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Simulation of one point")
    parser.add_argument("--dim_pcs", type=int, default=2,
                        help="Dimensionality of inputs")
    parser.add_argument("--nb_neurons", type=int, default=256,
                        help="Number of neurons")
    parser.add_argument("--dim_bbox", type=int, default=32,
                        help="Dimensionality of outputs")
    parser.add_argument("--num_scales", type=int, default=3,
                        help="Number of grid cell modules")
    parser.add_argument("--scale_fact", type=float, default=3/2,
                        help="Scale factor between GCs modules")
    parser.add_argument("--model", type=str, default='closed-load-polyae',
                        help="Type of model")   
    parser.add_argument("--load_id", type=int, default=2,
                        help="In case of load, id of the bbox to load")
    parser.add_argument("--input_amp", type=float, default=1.,
                        help="Amplitude of input")
    parser.add_argument('--input_dir', nargs='+', type=float, default=[2],
                        help="Direction of the input")
    parser.add_argument('--current_neurons', nargs='+',type=float,default=[0],
                        help="Neurons to recieve input current")
    parser.add_argument('--current_amp', type=float, default=0.,
                        help="Amplitude of the current input")
    parser.add_argument("--noise_amp", type=float, default=1.,
                        help="Amplitude of noise")
    parser.add_argument("--decoder_amp", type=float, default=0.2,
                        help="Amplitude of decoder matrix D")
    parser.add_argument("--thresh_amp", type=float, default=0.8,
                        help="Amplitude of the thresholds")                    
    parser.add_argument("--seed", type=int, default=666,
                        help="Random seed")
    parser.add_argument("--dir", type=str, default='./out/',
                        help="Directory to dump output")
    parser.add_argument("--plot", action='store_true', default=True,
                        help="Plot the results")
    parser.add_argument("--gif", action='store_true', default=False,
                        help="Generate a gif of the bbox")
    parser.add_argument("--save", action='store_true', default=False,
                        help="Save V, s, r and Th matrices")

    
    args = parser.parse_args()

    timestr = time.strftime("%Y%m%d-%H%M%S")
    code = ''.join(random.choice(string.ascii_letters) for i in range(5))
    name = timestr + "-" + code + "-" + args.model + "-pcs-" + str(args.dim_pcs) + "-bbox-" + str(args.dim_bbox) + "-n-" + str(args.nb_neurons)
    basepath = args.dir + name
    results = dict(datetime=timestr, basepath=basepath, args=vars(args))

    dbbox = args.dim_bbox
    n = args.nb_neurons

    # Construction of the path
    if args.dim_pcs == 1:
        path_type = 'ur'
    else:
        path_type = 'uspiral'
    p, dp, t, dt, time_steps = get_path(dpcs=args.dim_pcs, type=path_type)   

    ################################### GRID CELLS ###################################

    input_dim = 4
    print('Loading model...')
    model, D, G = get_model(input_dim, n, input_dim,connectivity=args.model, decod_amp=args.decoder_amp, thresh_amp=args.thresh_amp, load_id=args.load_id)
    
    # Construction of the current manipulation (noise + experiment)
    np.random.seed(seed=args.seed)
    I, b = get_current(input_dim, t, G, args.noise_amp, args.current_neurons, args.current_amp, vect='neuron')

    # Grid cells
    ay0 = np.zeros((time_steps))
    ay1 = np.zeros((time_steps))
    for s_idx in np.arange(args.num_scales):
        scale = (args.scale_fact)**s_idx

        # Construction of the input
        print('Codifying input...')
        Codifier = TorusCod()
        x, dx = Codifier.codify(p, dp, scale = scale)
        if scale == 1:
            ox = x
            dox = dx
        c = model.lamb*x + dx

        # Simulate/train the model
        print('Simulating model...')
        x0 = x[:,0]
        r0 = np.linalg.lstsq(D,x0+b[:,0]/model.lamb,rcond=None)[0]
        y0 = D @ r0 - b[:,0] / model.lamb        
        V0 = model.F @ x0 - G @ y0

        V, s, r = model.simulate(c, I, V0=V0, r0=r0, dt=dt, time_steps=time_steps)

        # Decode y
        decod = D @ r
        bias_corr = args.input_amp/(args.input_amp+0.5*(args.decoder_amp**2 - 1))
        gcy = bias_corr*decod - b / model.lamb
        gcdy = np.diff(gcy, prepend=gcy[:,0][:,None])/dt

        # Combine outputs # TODO 
        a0 = np.arctan2(gcy[1,:],gcy[0,:])
        a0[a0 < 0] += 2*np.pi
        a1 = np.arctan2(gcy[3,:],gcy[2,:])
        a1[a1 < 0] += 2*np.pi

        def g(x,scale):
            r = np.zeros_like(x)
            r[x > np.pi/scale] = 2*np.pi/scale
            r[x > 3*np.pi/scale] = 4*np.pi/scale
            return r
        
        a0 = 1/scale*a0
        a1 = 1/scale*a1
        ay0 = a0 + g(ay0-a0,scale)
        ay1 = a1 + g(ay1-a1,scale)

        # def g(x):
        #     r = np.zeros_like(x)
        #     r[x > np.pi] = 2*np.pi
        #     r[x > 3*np.pi] = 4*np.pi
        #     return r
        
        # ay0 = 1/scale*(a0 + g(scale*ay0 - a0))
        # ay1 = 1/scale*(a1 + g(scale*ay1 - a1))

    y = np.zeros_like(x)
    GCloc = np.array([ay0,ay1])/np.pi - 1
    y[0,:] = np.cos(ay0)
    y[1,:] = np.sin(ay0)
    y[2,:] = np.cos(ay1)
    y[3,:] = np.sin(ay1)
    y = 1/np.sqrt(2)*y
    dy = np.diff(y, prepend=y[:,0][:,None])/dt

    results['GC_tracking_error'] = np.mean(np.linalg.norm(y - ox,axis=0))
    results['GC_stracking_error'] = np.mean(np.linalg.norm(GCloc-p,axis=0))

    # Plot
    if args.plot:
        print('Generating neuroscience plot...')
        plot.plot_neuroscience(ox, y, V, s, t, basepath)

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
            plot.plot_2dspikebins(p, s, dt, 150, basepath, n_vect)
            plot.plot_2drfsth(D, x, p, basepath)


    ###################################### PLACE CELLS ################################

    decoder_amp = args.decoder_amp
    thresh_amp = args.thresh_amp - 0.25

    print('Loading model...')
    model, D, G = get_model(dbbox, n, dbbox,connectivity=args.model, decod_amp=decoder_amp, thresh_amp=thresh_amp, load_id=args.load_id)

    # Construction of the high dimensional embedding
    print('Embedding input...')
    Theta = get_basis(dbbox, dinput=x.shape[0], input_dir=args.input_dir, input_amp=args.input_amp, D=D, vect='random')

    # Embedd
    pcx = Theta @ y
    pcdx = Theta @ dy

    c = model.lamb*pcx + pcdx

    # Construction of the current manipulation (noise + experiment)
    np.random.seed(seed=args.seed)
    I, b = get_current(dbbox, t, G, args.noise_amp, args.current_neurons, args.current_amp, vect='neuron')

    # Simulate/train the model
    print('Simulating model...')
    x0 = pcx[:,0]
    r0 = np.linalg.lstsq(D,x0+b[:,0]/model.lamb,rcond=None)[0]
    y0 = D @ r0 - b[:,0] / model.lamb        
    V0 = model.F @ x0 - G @ y0

    V, s, r = model.simulate(c, I, V0=V0, r0=r0, dt=dt, time_steps=time_steps)

    # Decode y
    decod = D @ r
    bias_corr = args.input_amp/(args.input_amp+0.5*(args.decoder_amp**2 - 1))
    pcy = bias_corr*decod - b / model.lamb
    pcy_ld = Theta.T @ pcy # Reduce dimension 

    a0 = np.arctan2(pcy_ld[1,:],pcy_ld[0,:])
    a0[a0 < 0] += 2*np.pi
    a1 = np.arctan2(pcy_ld[3,:],pcy_ld[2,:])
    a1[a1 < 0] += 2*np.pi
    PCloc = np.array([a0,a1])/np.pi - 1

    # Save results 
    print('Saving results...')
    results['y_end'] = y[:,-1].tolist()
    results['PC_tracking_error'] = np.mean(np.linalg.norm(pcy_ld - y,axis=0))
    results['PC_stracking_error'] = np.mean(np.linalg.norm(PCloc-GCloc, axis=0))
    results['Global_tracking_error'] = np.mean(np.linalg.norm(pcy_ld - ox, axis=0))
    results['Global_stracking_error'] = np.mean(np.linalg.norm(PCloc-p, axis=0))

    # PCs
    # active_list = np.any(s,axis=1)
    # pcs_list = np.where(active_list)[0]
    # npcs = pcs_list.shape[0]

    # # FRs
    # ft = 1
    # m = int(ft/dt)
    # filter = np.ones(m)
    # fr = np.apply_along_axis(lambda m: np.convolve(m, filter, mode='same'), axis=1, arr=s)
    # maxfr = np.max(fr[active_list,:], axis=1)
    # if np.max(maxfr) <= ft/1e-3:
    #     print('1 spike/ 1 ms asserted')
    # meanfr = np.mean(fr[active_list,:], axis=1)

    # results['perpcs'] = npcs/n
    # results['pcsidx'] = pcs_list.tolist()
    # results['maxfr'] = maxfr.tolist()
    # results['meanfr'] = meanfr.tolist()

    # if args.save:
    #     np.savetxt("%s-Th.csv" % basepath, Theta, fmt='%.3e')
    #     results['Th'] = "%s-Th.csv" % name
    #     # np.savetxt("%s-V.csv" % basepath, V)
    #     # results['V'] = "%s-V.csv" % name
    #     np.savetxt("%s-s.csv" % basepath, s, fmt='%i')
    #     results['s'] = "%s-s.csv" % name
    #     # np.savetxt("%s-r.csv" % basepath, r)
    #     # results['r'] = "%s-r.csv" % name
    
    filepath = "%s.json" % basepath
    with open(filepath, "w") as file_handle:
        json.dump(results, file_handle, indent=4)

    # Plot
    if args.plot:
        basepath = basepath + '-PCS'
        print('Generating neuroscience plot...')
        plot.plot_neuroscience(ox, pcy_ld, V, s, t, basepath + 'G')
        plot.plot_neuroscience(y, pcy_ld, V, s, t, basepath)

        if args.dim_pcs == 1:
            print('Generating 1drfs plot...')

            plot.plot_1drfs(p, r, dt, basepath, pad=0)
            plot.plot_1dspikebins(p, s, 25, basepath, pad=0)
            plot.plot_1drfsth(D, pcx, p, basepath, pad=0)
        
        if args.dim_pcs == 2:
            print('Generating 2drfs plot...')

            if n > 49:
                n_vect = np.where(np.any(s,axis=1))[0]
                n_vect = n_vect[:49]
            else: 
                n_vect = np.arange(n)
           
            plot.plot_2drfs(p, r, dt, basepath, n_vect)
            plot.plot_2dspikebins(p, s, dt, 150, basepath, n_vect)
            plot.plot_2drfsth(D, pcx, p, basepath)