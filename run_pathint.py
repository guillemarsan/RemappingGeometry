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
    parser.add_argument("--dim_pcs", type=int, default=1,
                        help="Dimensionality of space")
    parser.add_argument('--path_seed',type=int, default=0,
                        help="Random seed for the path in case of random")
    
    parser.add_argument("--encoding", type=str, default='rotation',
                        help='Determines the type of encoder between rotation, parallel and flexible')
    parser.add_argument("--dim_bbox", type=int, default=8,
                        help="Dimensionality of latent space")
    parser.add_argument('--env', type=int, default=2,
                        help="Environment id")
    parser.add_argument('--embedding_sigma', type=float, default=-1,
                        help="Variance in case of biased embedding (not -1)")
    parser.add_argument("--input_amp", type=float, default=1.,
                        help="Amplitude of input")
    parser.add_argument("--input_scale", action='store_true', default=True,
                        help="Scale the input by sqrtdbbox") 
    
    
    parser.add_argument("--nb_neurons", type=int, default=512,
                        help="Number of neurons")
    parser.add_argument("--model", type=str, default='randclosed-load-polyae',
                        help="Type of model")
    parser.add_argument("--rnn", action='store_true', default=True,
                        help="Either rnn or feed-forward")
    parser.add_argument("--simulate", type=str, default='pathint_one',
                        help='Determines the type of simulation between integrator (pathint) and one spike (one)')
    parser.add_argument("--spikeone", action='store_true', default=True,
                        help="Integrator with only one neurons spiking per timestep")  
    parser.add_argument('--conn_seed',type=int, default=0,
                        help="Random seed for the connectivity in case of random")
    parser.add_argument("--load_id", type=int, default=1,
                        help="Id of the connectivity in case of load") 
    parser.add_argument("--decoder_amp", type=float, default=0.3,
                        help="Amplitude of decoder matrix D")
    parser.add_argument("--thresh_amp", type=float, default=0.6,
                        help="Amplitude of the thresholds")    
    parser.add_argument('--lognor_seed',type=int, default=0,
                        help="The thresholds are taken from a lognormal distribution if not 0") 
    parser.add_argument('--lognor_sigma',type=float, default=0.2,
                        help="The variance of the lognormal for sampling thresholds")
    
    parser.add_argument("--noise_seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--noise_amp", type=float, default=0.,
                        help="Amplitude of noise")
    
    parser.add_argument('--current_per', type=float,default=0.,
                        help="Percentage of neurons to recieve input current")
    parser.add_argument('--current_seed', type=int,default=0,
                        help="Random seed for neurons to recieve input current")
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
                        help="Save V, s, r and Th matrices")

    
    args = parser.parse_args()

    timestr = time.strftime("%Y%m%d-%H%M%S")
    code = ''.join(random.choice(string.ascii_letters) for i in range(5))
    name = timestr + "-" + code + "-" + args.model + "-pcs-" + str(args.dim_pcs) + "-bbox-" + str(args.dim_bbox) + "-n-" + str(args.nb_neurons)
    basepath = args.dir + name

    dbbox = args.dim_bbox
    n = args.nb_neurons

    print('Loading model...')
    model, D, G = get_model(dbbox ,n, dbbox,connectivity=args.model, decod_amp=args.decoder_amp, 
                    thresh_amp=args.thresh_amp, load_id=args.load_id, conn_seed=args.conn_seed, 
                    lognor_seed=args.lognor_seed, lognor_sigma=args.lognor_sigma, rnn=args.rnn)

    # Construction of the path
    args.path_tmax = 30 #s
    if args.dim_pcs == 1:
        args.path_type = 'ur'
    else:
        args.path_type = 'usnake'

    # Load gamma path
    print('Loading path...')
    p, dp, t, dt, time_steps = get_path(dpcs=args.dim_pcs, type=args.path_type, tmax=args.path_tmax, path_seed=args.path_seed) 

    results = dict(datetime=timestr, basepath=basepath, args=vars(args))
    
    if args.encoding == 'rotation':
        # Only position variables
        g = p
        dg = dp

        # Angle encoding to semicircles
        print('Encoding input...')
        Encoder = AngleEncoder()
        k, dk = Encoder.encode(g, dg)
        dinput = k.shape[0]

        # Construction of Theta(e)
        print('Embedding input...')
        Theta = get_embedding(dbbox, dinput=dinput, env=args.env, variance=args.embedding_sigma)  
        
        # Embedd
        x = Theta @ k
        dx = Theta @ dk
    else:
        # Also environment variables
        dim_e = int((dbbox - 2*args.dim_pcs)/2)
        e, de, eofp = get_pathe(p, dim_e, args.env, flexible=args.encoding=='flexible', variance=args.embedding_sigma)
        g = np.vstack([p,e])
        dg = np.vstack([dp,de])

        # Angle encoding to semicircles
        print('Encoding input...')
        Encoder = AngleEncoder()
        k, dk = Encoder.encode(g, dg)
        dinput = k.shape[0]
    
        # Embedd (Theta = Id)
        x = k
        dx = dk

    # Scale of input
    if args.input_scale:
        x = np.sqrt(dbbox)*x
        dx = np.sqrt(dbbox)*dx
    else:
        x = args.input_amp*x
        dx = args.input_amp*dx

    # Noise
    I, b = get_noise(dbbox, t, G, noise_amp=args.noise_amp, noise_seed=args.noise_seed)
    
    # Current manipulation
    I += get_current(n, t, current_per=args.current_per, current_amp=args.current_amp, current_seed=args.current_seed)

    # Bias correction for D
    input_amp = np.sqrt(dbbox)*args.input_amp if args.input_scale else args.input_amp
    bias_corr = input_amp/(input_amp+0.5*(args.decoder_amp - args.thresh_amp))

    # Simulate the model
    print('Simulating model...')
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
        V, s, r, x_hat = model.simulate_one(c, I, V0=V0, r0=r0, dt=dt, time_steps=time_steps)
    else:
        c = model.lamb*x + dx
        V, s, r, x_hat = model.simulate(c, I, V0=V0, r0=r0, dt=dt, time_steps=time_steps)


    # Decode
    x_hat = bias_corr*x_hat

    k_hat = Theta.T @ x_hat if args.encoding == 'rotation' else x_hat
    g_hat = Encoder.decode(k_hat)


    # Save results 
    print('Saving results...')
    results['x_error'] = np.mean(np.linalg.norm(x_hat - x, axis=0))
    results['k_error'] = np.mean(np.linalg.norm(k_hat - k, axis=0))
    results['g_error'] = np.mean(np.linalg.norm(g_hat - g, axis=0))
    if args.encoding != 'rotation':
        results['p_error'] = np.mean(np.linalg.norm(g_hat[:args.dim_pcs,:] - g[:args.dim_pcs,:], axis=0))
        results['e_error'] = np.mean(np.linalg.norm(g_hat[args.dim_pcs:,:] - g[args.dim_pcs:,:], axis=0))

    # PCs
    active_list = np.any(s,axis=1)
    pcs_list = np.where(active_list)[0]
    npcs = pcs_list.shape[0]
    results['perpcs'] = npcs/n
    results['pcsidx'] = pcs_list.tolist()

    results['nb_steps'] = time_steps
    results['dt'] = dt

    # FRs
    if args.compute_fr:
        print('Computing firing rates...')
        ft = 1
        m = int(ft/dt)
        filter = np.ones(m)/ft
        fr = np.apply_along_axis(lambda m: np.convolve(m, filter, mode='same'), axis=1, arr=s)
        maxfr = np.max(fr[active_list,:], axis=1)
        if np.max(maxfr) <= ft/1e-3:
            print('1 spike/ 1 ms asserted')
        meanfr = np.mean(fr[active_list,:], axis=1)

    
        results['maxfr'] = maxfr.tolist()
        results['meanfr'] = meanfr.tolist()

    if args.save:
        if args.encoding == 'rotation':
            np.savetxt("%s-Th.csv" % basepath, Theta, fmt='%.3e')
        
        spike_times = np.argwhere(s)
        np.savetxt("%s-stimes.csv" % basepath, spike_times, fmt='%i')
        
        #np.savetxt("%s-D.csv" % basepath, D, fmt='%.3e')
    
    filepath = "%s.json" % basepath
    with open(filepath, "w") as file_handle:
        json.dump(results, file_handle, indent=4)

    # Plot
    if args.plot:

        if args.encoding != 'rotation':
            print('Generating (p,e) plot...')
            plot.plot_pe(p, eofp, e, t, basepath)

        if n > 49:
                n_vect = np.where(np.any(s,axis=1))[0]
                n_vect = n_vect[:49]
        else: 
            n_vect = np.arange(n)

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
            plot.plot_1dspikebins(p, s, 25, basepath, n_vect)
            plot.plot_1drfsth(D, x, p, basepath)
        
        if args.dim_pcs == 2:
            print('Generating 2drfs plot...')
           
            plot.plot_2drfs(p, r, basepath, n_vect)
            plot.plot_2dspikebins(p, s, dt, 100, basepath, n_vect)
            plot.plot_2drfsth(D, x, p, basepath)

    if args.gif and dbbox == 2:
        print('Generating gif...')
        plot.plot_1danimbbox(x, x_hat, model.F, G, model.T, basepath, plotx=(args.model == 'randae' or args.model == 'polyae'))