import argparse
import time
import json
import string
import random

import numpy as np
from convexsnn.AngleEncoder import AngleEncoder



from convexsnn.embedding import get_embedding
from convexsnn.network import get_model
from convexsnn.current import get_noise, get_current
from convexsnn.path import get_path, get_pathe
import convexsnn.plot as plot

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Simulation of one point")
    parser.add_argument("--dim_pcs", type=int, default=2,
                        help="Dimensionality of space")
    parser.add_argument("--path_type", type=str, default='usnake',
                        help='Type of path the animal does')
    parser.add_argument('--path_tmax', type=float, default=30.,
                        help="Length of the path in seconds")
    parser.add_argument('--path_seed',type=int, default=0,
                        help="Random seed for the path in case of random")
    
    parser.add_argument("--encoding", type=str, default='gridcells',
                        help='Determines the type of encoder between rotation, parallel and flexible')
    parser.add_argument("--dim_bbox", type=int, default=8,
                        help="Dimensionality of latent space")
    parser.add_argument('--env', type=int, default=0,
                        help="Environment id")
    parser.add_argument('--embedding_sigma', type=float, default=-1,
                        help="Variance in case of biased embedding (not -1)")
    parser.add_argument("--input_amp", type=float, default=1.,
                        help="Amplitude of input")
    parser.add_argument("--input_sepnorm",action='store_true', default=False,
                        help="Normalize separate the first 2*dpcs dimensions. Only for parallel, flexible")
    parser.add_argument("--input_scale", action='store_true', default=True,
                        help="Scale the input by sqrtdbbox") 
    
    
    parser.add_argument("--nb_neurons", type=int, default=64,
                        help="Number of neurons")
    parser.add_argument("--model", type=str, default='randclosed-load-polyae',
                        help="Type of model")
    parser.add_argument("--rnn", action='store_true', default=True,
                        help="Either rnn or feed-forward")
    parser.add_argument("--simulate", type=str, default='minimization',
                        help='Determines the type of simulation between integrator (pathint) and one spike (one)')
    parser.add_argument('--conn_seed',type=int, default=0,
                        help="Random seed for the connectivity in case of random")
    parser.add_argument("--load_id", type=int, default=0,
                        help="Id of the connectivity in case of load")
    parser.add_argument("--model_conj",type=str, default='CM',
                        help="Normalize separate each couple of dimensions: M, C, CM, CC") 
    parser.add_argument("--decoder_amp", type=float, default=1,
                        help="Amplitude of decoder matrix D")
    parser.add_argument("--thresh_amp", type=float, default=1,
                        help="Amplitude of the thresholds")    
    parser.add_argument('--lognor_seed',type=int, default=0,
                        help="The thresholds are taken from a lognormal distribution if not 0") 
    parser.add_argument('--lognor_sigma',type=float, default=0.2,
                        help="The variance of the lognormal for sampling thresholds")
    
    parser.add_argument("--noise_seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--noise_amp", type=float, default=0.,
                        help="Amplitude of noise")
    
    parser.add_argument('--tagging_sparse', type=float,default=0.,
                        help="Probability of neurons to express opsin")
    parser.add_argument('--tagged_idx', nargs='+', type=int, default=[],
                        help="Neurons already tagged for inhibition")
    parser.add_argument('--current_amp', type=float, default=0,
                        help="Amplitude of the current input")
    
    parser.add_argument("--dir", type=str, default='./out/',
                        help="Directory to dump output")
    parser.add_argument("--compute_fr", action='store_true', default=False,
                        help="Compute rough meanfr for quick check")
    parser.add_argument("--plot", action='store_true', default=False,
                        help="Plot the results")
    parser.add_argument("--gif", action='store_true', default=False,
                        help="Generate a gif of the bbox")
    parser.add_argument("--save", action='store_true', default=True,
                        help="Save s and r matrices")
    parser.add_argument("--save_input", action='store_true', default=False,
                        help="Save Th and x matrices")

    
    args = parser.parse_args()

    timestr = time.strftime("%Y%m%d-%H%M%S")
    code = ''.join(random.choice(string.ascii_letters) for i in range(5))
    name = timestr + "-" + code + "-" + args.model + "-pcs-" + str(args.dim_pcs) + "-bbox-" + str(args.dim_bbox) + "-n-" + str(args.nb_neurons)
    basepath = args.dir + name

    dbbox = args.dim_bbox
    n = args.nb_neurons

    print('Loading model...')
    sepnorm = None if len(args.model_conj) == 1 else args.dim_pcs*2
    model, D, G = get_model(dbbox ,n, dbbox,connectivity=args.model, decod_amp=args.decoder_amp, 
                    thresh_amp=args.thresh_amp, load_id=args.load_id, conn_seed=args.conn_seed, 
                    lognor_seed=args.lognor_seed, lognor_sigma=args.lognor_sigma, rnn=args.rnn, conj=args.model_conj,
                    sepnorm=sepnorm)

    ## CONSTRUCTION OF P
    # Load gamma path
    print('Loading path...')
    p, dp, t, dt, time_steps = get_path(dpcs=args.dim_pcs, type=args.path_type, tmax=args.path_tmax, path_seed=args.path_seed) 

    results = dict(datetime=timestr, basepath=basepath, args=vars(args))
    
    ## CONSTRUCTION OF P,Q
    if args.encoding in {'parallel', 'flexible', 'sensory', 'flexibleGP', 'sensoryGP', 'flexibler'}:
        # Also environment variables
        dim_e = int((dbbox - 2*args.dim_pcs)/2) if args.encoding in {'parallel', 'flexible', 'flexibleGP', 'flexibler'} else int(dbbox/2)

        variab = 'l' if args.encoding == 'parallel' else \
                ('m' if args.encoding in {'flexibleGP', 'sensoryGP'} else \
                ('h' if args.encoding in {'flexible, sensory'} else 'r'))
        e, de, eofp = get_pathe(p, dim_e, args.env, dt, variability = variab, variance=args.embedding_sigma)

        if args.encoding in {'parallel', 'flexible', 'flexibleGP', 'flexibler'}:
            g = np.vstack([p,e])
            dg = np.vstack([dp,de])
        else: 
            g = e
            dg = de
    else:
        # Only position variables
        g = p
        dg = dp

    ## CONSTRUCTION OF Z
    # Angle encoding to semicircles
    print('Encoding input...')
    modules = int(dbbox/(2*args.dim_pcs)) if args.encoding == 'gridcells' else 1
    Encoder = AngleEncoder()
    k, dk = Encoder.encode(g, dg, modules=modules)
    dinput = k.shape[0]

    ## CONSTRUCTION OF Y
    # Construction of Theta(e)
    print('Embedding input...')
    if args.encoding in {'rotation', 'gridcells'}:
        Theta = get_embedding(dbbox, dinput=dinput, env=args.env, variance=args.embedding_sigma, sphere=args.encoding=='rotation') 
    else:
        Theta = np.eye(dinput)
    
    # Embedd
    x = Theta @ k
    dx = Theta @ dk

    # Scale of input
    if args.input_sepnorm:
        x[:args.dim_pcs*2,:] = 1/np.sqrt(2) * x[:args.dim_pcs*2,:]/np.linalg.norm(x[:args.dim_pcs*2,:],axis=0)  
        x[args.dim_pcs*2:,:] = 1/np.sqrt(2) * x[args.dim_pcs*2:,:]/np.linalg.norm(x[args.dim_pcs*2:,:],axis=0) 
        if args.simulate != 'minimization':
            dx[:args.dim_pcs*2,:] = 1/np.sqrt(2) * dx[:args.dim_pcs*2,:]/np.linalg.norm(dx[:args.dim_pcs*2,:],axis=0)  
            dx[args.dim_pcs*2:,:] = 1/np.sqrt(2) * dx[args.dim_pcs*2:,:]/np.linalg.norm(dx[args.dim_pcs*2:,:],axis=0) 

    if args.input_scale:
        x = np.sqrt(dbbox)*x
        dx = np.sqrt(dbbox)*dx
    else:
        x = args.input_amp*x
        dx = args.input_amp*dx

    ## NOISE AND CURRENT MANIPULATION
    if args.simulate != 'minimization':
        # Noise
        I, b = get_noise(dbbox, t, G, noise_amp=args.noise_amp, noise_seed=args.noise_seed)
        
        # Current manipulation
        I += get_current(n, t, tagged_idx=args.tagged_idx, current_amp=args.current_amp)
    else:
        b = np.zeros_like(x)
        # Current manipulation
        mask = np.zeros_like(model.T)
        mask[args.tagged_idx] = 1
        model.T = model.T - mask * args.current_amp

    ## SIMULATION
    # Bias correction for D
    input_amp = np.sqrt(dbbox)*args.input_amp if args.input_scale else args.input_amp
    bias_corr = input_amp/(input_amp+0.5*(args.decoder_amp - args.thresh_amp))

    # Simulate the model
    print('Simulating model...')
    if args.simulate != 'minimization':
        x0 = x[:,0]
        r0 = np.linalg.lstsq(bias_corr*D,x0+b[:,0]/model.lamb,rcond=None)[0]
        x_hat0 = D @ r0 - b[:,0] / model.lamb   
        V0 = model.F @ x0 - G @ x_hat0

    decoder = lambda r, i: D @ r - b[:,i] / model.lamb
    if args.simulate == 'pathint_one':
        V, s, r, x_hat = model.simulate_pathint_one(dx, I, decoder, x0=x0, V0=V0, r0=r0, dt=dt, time_steps=time_steps)
    elif args.simulate == 'pathint':
        V, s, r, x_hat = model.simulate_pathint(dx, I, decoder, x0=x0, V0=V0, r0=r0, dt=dt, time_steps=time_steps)
    elif args.simulate == 'one':
        c = model.lamb*x + dx
        V, s, r = model.simulate_one(c, I, V0=V0, r0=r0, dt=dt, time_steps=time_steps)
        x_hat = D @ r - b / model.lamb
    elif args.simulate == 'minimization':
        r = model.simulate_minimization(x)
        x_hat = D @ r - b / model.lamb
    else:
        c = model.lamb*x + dx
        V, s, r = model.simulate(c, I, V0=V0, r0=r0, dt=dt, time_steps=time_steps)
        x_hat = D @ r - b / model.lamb

    ## DECODING
    # Decode
    x_hat = bias_corr*x_hat
    k_hat = Theta.T @ x_hat if args.encoding in {'rotation', 'gridcells'} else x_hat
    k_hat = k_hat/np.sqrt(dbbox) if args.input_scale else k_hat/args.input_amp
    g_hat = Encoder.decode(k_hat, modules=modules)

    ## RESULTS
    # Save results 
    print('Saving results...')
    results['x_error'] = np.mean(np.linalg.norm(x_hat - x, axis=0))
    results['k_error'] = np.mean(np.linalg.norm(k_hat - k, axis=0))
    results['g_error'] = np.mean(np.linalg.norm(g_hat - g, axis=0))
    if args.encoding in {'parallel', 'flexible', 'flexibleGP', 'flexibler'}:
        results['p_error'] = np.mean(np.linalg.norm(g_hat[:args.dim_pcs,:] - g[:args.dim_pcs,:], axis=0))
        results['e_error'] = np.mean(np.linalg.norm(g_hat[args.dim_pcs:,:] - g[args.dim_pcs:,:], axis=0))

    # PCs
    active_list = np.any(s,axis=1) if args.simulate != 'minimization' else np.any(r > 0,axis=1)
    pcs_list = np.where(active_list)[0]
    npcs = pcs_list.shape[0]
    results['peractive'] = npcs/n
    results['activeidx'] = pcs_list.tolist()

    results['nb_steps'] = time_steps
    results['dt'] = dt

    # FRs
    if args.compute_fr:
        print('Computing firing rates...')
        ft = 1
        if args.simulate != 'minimization':
            m = int(ft/dt)
            filter = np.ones(m)/ft
            fr = np.apply_along_axis(lambda m: np.convolve(m, filter, mode='same'), axis=1, arr=s)
            maxfr = np.max(fr[active_list,:], axis=1)
            if np.max(maxfr) <= ft/1e-3:
                print('1 spike/ 1 ms asserted')
            meanfr = np.mean(fr[active_list,:], axis=1)
        else:
            maxfr = np.max(r[active_list,:],axis=1)
            meanfr = np.mean(r[active_list,:],axis=1)
            if np.max(maxfr) <= ft/1e-3:
                print('1 spike/ 1 ms asserted')

    
        results['maxfr'] = maxfr.tolist()
        results['meanfr'] = meanfr.tolist()

    if args.save:
        if args.simulate != 'minimization':
            spike_times = np.argwhere(s)
            np.savetxt("%s-stimes.csv" % basepath, spike_times, fmt='%i')
        else:
            np.savetxt("%s-rates.csv" % basepath, r, fmt='%.5e')
    if args.save_input:
        if args.encoding in {'rotation', 'gridcells'}:
            np.savetxt("%s-Th.csv" % basepath, Theta, fmt='%.3e')
        np.savetxt("%s-x.csv" % basepath, x, fmt='%.3e')
        np.savetxt("%s-xhat.csv" % basepath, x_hat, fmt='%.3e')

        if args.encoding != 'gridcells':
            np.savetxt("%s-g.csv" % basepath, g, fmt='%.3e')
            np.savetxt("%s-ghat.csv" % basepath, g_hat, fmt='%.3e')
        else:
            g = np.vstack([Encoder.decode(x[:2,:]), Encoder.decode(x[2:,:])])
            g_hat = np.vstack([Encoder.decode(x_hat[:2,:]), Encoder.decode(x_hat[2:,:])])
            np.savetxt("%s-g.csv" % basepath, g, fmt='%.3e')
            np.savetxt("%s-ghat.csv" % basepath, g_hat, fmt='%.3e')

        
        #np.savetxt("%s-D.csv" % basepath, D, fmt='%.3e')
    
    filepath = "%s.json" % basepath
    with open(filepath, "w") as file_handle:
        json.dump(results, file_handle, indent=4)

    # Plot
    if args.plot:

        if args.encoding in {'parallel', 'flexible', 'flexibleGP', 'flexibler', 'sensory', 'sensoryGP'}:
            print('Generating (p,e) plot...')
            plot.plot_pe(p, eofp, e, t, basepath)

        if n > 49:
            n_vect = np.arange(49)
        else: 
            n_vect = np.arange(n)
            
        if args.simulate != 'minimization':
            print('Generating neuroscience plot...')
            plot.plot_neuroscience(x, x_hat, V, s, t, basepath, n_vect, T=model.T[0])

        if dbbox == 2 or dbbox ==3:
            print('Generating bounding box plot...')
            if dbbox == 2:
                plot.plot_1dbbox(x[:,-1], x_hat[:,-1:], model.F, G, model.T, basepath, plotx=(args.model == 'randae' or args.model == 'polyae'))
            else:
                plot.plot_2dbboxproj(model.F, G, model.T, args.input_amp, basepath)

        if args.dim_pcs == 1:
            print('Generating 1drfs plot...')

            plot.plot_1drfs(p, r, dt, basepath, n_vect)
            if args.simulate != 'minimization': plot.plot_1dspikebins(p, s, 25, basepath, n_vect)
            plot.plot_1drfsth(D, x, p, basepath)
        
        if args.dim_pcs == 2:
            print('Generating 2drfs plot...')
           
            plot.plot_2drfs(p, r, basepath, n_vect)
            if args.simulate != 'minimization': plot.plot_2dspikebins(p, s, dt, 100, basepath, n_vect)
            plot.plot_2drfsth(D, x, p, basepath)

    if args.gif and dbbox == 2:
        print('Generating gif...')
        plot.plot_1danimbbox(x, x_hat, model.F, G, model.T, basepath, plotx=(args.model == 'randae' or args.model == 'polyae'))