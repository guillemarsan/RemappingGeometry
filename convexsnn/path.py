import scipy
import scipy.signal, scipy.ndimage
import numpy as np
import time
from utils import compute_meshgrid
from scipy.stats import ortho_group



def get_path(dpcs, type, tmax, path_seed=0):
    
    dt = 0.0001
    time_steps = int(tmax * 1/dt)
    t = np.arange(time_steps)*dt

    ones = np.ones((dpcs,time_steps))

    if type == 'constant':
        p = 0.5*ones
        p[0,:] = -p[0,:]
        dp = 0*ones

    elif type == 'ur':
        start = -0.95
        finish = 0.95
        v = (finish - start)/tmax
        p = start + v*t*ones
        dp = v*ones

    elif type == 'uspiral':
        r0 = 0.95
        nint = r0*(tmax - t)/tmax
        a = 2 # half circles

        p = nint*np.cos(a*t)*ones
        p[1,:] = nint*np.sin(a*t)*ones[1,:]
        dp = (-r0/tmax*np.cos(a*t) - nint*a*np.sin(a*t))*ones
        dp[1,:] = (-r0/tmax*np.sin(a*t) + nint*a*np.cos(a*t))*ones[1,:]

    elif type == 'usnake':
        xi = 0.9
        xe = -0.9
        bars = 17

        bar_length = xi - xe
        drop_length = bar_length/(bars-1)
        path_length = bar_length*(bars+1)

        vel = path_length/time_steps
        bar_timsteps = int(bar_length/vel)
        drop_timsteps = int(drop_length/vel)

        p = np.zeros((dpcs,time_steps))
        dp = np.zeros((dpcs,time_steps))
        i = 0
        for bar in np.arange(bars):
            # Bar
            dir = 1 if bar % 2 == 0 else -1
            if i == 0:
                p[0,i:i+bar_timsteps] = np.linspace(xe,xe + dir*bar_length,bar_timsteps)
                p[1,i:i+bar_timsteps] = xi
            else:
                p[0,i:i+bar_timsteps] = np.linspace(p[0,i-1],p[0,i-1] + dir*bar_length,bar_timsteps)
                p[1,i:i+bar_timsteps] = p[1,i-1]
            dp[0,i:i+bar_timsteps] = dir*vel/dt 
            dp[1,i:i+bar_timsteps] = 0
            i = i+bar_timsteps

            if bar < bars-1:
                # Drop
                p[0,i:i+drop_timsteps] = p[0,i-1]
                p[1,i:i+drop_timsteps] = np.linspace(p[1,i-1],p[1,i-1] - drop_length, drop_timsteps)
                dp[0,i:i+drop_timsteps] = 0 
                dp[1,i:i+drop_timsteps] = -vel/dt
                i = i+drop_timsteps

        p[:,i:] = np.expand_dims(p[:,i-1],axis=1)

    elif type== '2Drandwalk':
        np.random.seed(seed=int(path_seed))

        turns = np.random.randint(time_steps/10,time_steps/2)
        acc_times = np.sort(np.random.randint(time_steps, size=(2*turns,)))
        acc_mag = np.random.uniform(-20,20,(2,turns))
        acc = np.zeros((2,time_steps))
        for turn in range(turns):
            acc[:,acc_times[2*turn]:acc_times[2*turn+1]] = np.expand_dims(acc_mag[:,turn],axis=1)

        p = np.zeros((dpcs,time_steps))
        p[:,0] = np.random.uniform(-1,1,(2,))
        dp = np.zeros((dpcs,time_steps))
        for ts in range(time_steps-1):
            dp[:,ts+1] = dp[:,ts] + dt*acc[:,ts+1]
            p[:,ts+1] = p[:,ts] + dt*dp[:,ts+1]

            for d in range(2):
                if p[d,ts+1] > 1:
                    p[d,ts+1] = 1
                    dp[d,ts+1] = 0
                elif p[d,ts+1] < -1:
                    p[d,ts+1] = -1
                    dp[d,ts+1] = 0
        print(np.max(dp))    

    elif type == 'grid':
        npoints = 400
        p = compute_meshgrid(1, npoints, dpcs)
        dp = np.zeros_like(p)
        dt = 1
        time_steps = npoints
        t = np.arange(time_steps)*dt

    return p, dp, t, dt, time_steps

def get_pathe(p, dim_e, env, dt, flexible=False, variance=-1):

    def gram(x, sigma=5, l=2):
        condensed_dist = scipy.spatial.distance.pdist(x)
        dist = scipy.spatial.distance.squareform(condensed_dist)
        gram = sigma**2 * np.exp(-1 * (dist ** 2) / (2*(l**2)))
        return gram
    
    def mesh(num_bins, dim_pcs):

        if dim_pcs == 1:
            points = np.linspace(-1,1,num_bins).reshape(1,-1)
            step = (2/(num_bins-1))/2
        else:
            sqrtnum_bins = int(np.sqrt(num_bins))
            points = np.linspace(-1,1,sqrtnum_bins)
            step = (2/(sqrtnum_bins-1))/2
            points = np.array(np.meshgrid(points,points)).reshape(2,-1)

        return step, points

    
    dim_pcs = p.shape[0]
    sqrtnum_bins = 20
    num_bins = sqrtnum_bins**2

    step, points = mesh(num_bins, dim_pcs)
        
    eofp = np.ones((dim_e, p.shape[1]))
    np.random.seed(env)
    if not flexible: 
        if variance == -1:
            nu = np.random.uniform(-1,1,(dim_e,1))
            eofp = eofp * nu
        else:
            for i in np.arange(dim_e): 
                nu = 2
                while nu > 1 or nu < -1:
                    nu = np.random.normal(0,variance)
                eofp[i,:] = np.ones(points.shape[1])*nu
    else:
        ftype = 'norm'
        if ftype == 'norm':
            c = np.random.uniform(-1,1,(dim_e,1))
            vel = 0.25*np.sqrt(dim_e) if variance == -1 else variance*0.25*np.sqrt(dim_e) #0.25
            th = ortho_group.rvs(dim_e)[:,:dim_pcs] if dim_e > 1 else np.random.choice([1,-1], (1,1))
            eofp = vel*th @ points + c
        else:
            covar = gram(points.T)
            for i in np.arange(dim_e): 
                eofp[i,:] = np.random.multivariate_normal(np.zeros(points.shape[1]),variance*covar)

    eofp = (eofp + 1) % 2 - 1
    if dim_pcs == 2: eofp = eofp.reshape((-1, sqrtnum_bins,sqrtnum_bins))

    # upsample cause gram matrix is limiting
    upscale = False
    factor = 5 if upscale else 1
    if upscale: 
        factor = 5
        eofpf = np.zeros((eofp.shape[0], factor*eofp.shape[1])) if dim_pcs == 1 else np.zeros((eofp.shape[0], factor*eofp.shape[1], factor*eofp.shape[2]))
        for i in np.arange(dim_e):
            if dim_pcs == 1:
                eofpf[i,:] = np.repeat(eofp[i,:], factor)
                filter = np.ones(factor)/factor
                eofpf[i,:] = scipy.ndimage.convolve1d(eofpf[i,:], filter, mode='nearest')
                step_up, _ = mesh(factor*num_bins, dim_pcs)
            else:
                eofpf[i,:,:] = np.repeat(np.repeat(eofp[i,:,:], factor, axis=0), factor, axis=1)
                filter = np.ones((factor,factor))/(factor**2)
                t0 = time.time()
                eofpf[i,:,:] = scipy.signal.convolve2d(eofpf[i,:,:], filter, mode='same', boundary='symm')
                t1 = time.time()
                print(t1-t0)
                step_up, _ = mesh(factor**2*num_bins, dim_pcs)
        eofp = eofpf
    else: step_up, _ = mesh(factor**2*num_bins, dim_pcs)

    # Get path intersection
    # t0 = time.time()
    # eofplin = eofp.reshape(-1,)
    # for j in np.arange(eofplin.shape[0]):
    #     dist = np.max(np.abs(np.expand_dims(points_up[:,j],-1)-p), axis=0) # Manhattan distance
    #     de[i,np.argwhere(dist < step_up)] = eofplin[j]
    # t1 = time.time()
    # print(t1-t0)

    e = np.zeros((dim_e, p.shape[1]))
    de = np.zeros((dim_e, p.shape[1]))
    for i in np.arange(dim_e):
        if dim_pcs == 1:
            t0 = time.time()
            origin = np.array([-1])[:,np.newaxis]
            idcs = (np.abs(p - origin) // (2*step_up)).astype(int)
            e[i,:] = eofp[i,idcs]
            t1 = time.time()
            print(t1-t0)

        else:
            t0 = time.time()
            origin = np.array([-1,1])[:,np.newaxis]
            idcs = (np.abs(p - origin) // (2*step_up)).astype(int)
            e[i,:] = eofp[i,idcs[0,:],idcs[1,:]]
            t1 = time.time()
            print(t1-t0)

        aux = np.diff(e[i,:])/dt
        de[i,:] = np.append(aux,aux[-1])
        print(np.max(de[i,:]))
        print(np.min(de[i,:]))
    return e, de, eofp