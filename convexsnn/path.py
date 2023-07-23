import scipy
import scipy.signal, scipy.ndimage
import numpy as np
import time

dt = 0.0001

def get_path(dpcs, type, tmax, path_seed=0):
    
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
    return p, dp, t, dt, time_steps

def get_pathe(p, dim_e, env, flexible=False, variance=-1):

    def gram(x):
        sigma = 0.75
        condensed_dist = scipy.spatial.distance.pdist(x)
        dist = scipy.spatial.distance.squareform(condensed_dist)
        gram = np.exp(-1 * (dist ** 2) / (2*sigma))
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
    sqrtnum_bins = 21
    num_bins = sqrtnum_bins**2

    step, points = mesh(num_bins, dim_pcs)
    
    covar = gram(points.T)

    e = np.zeros((dim_e, p.shape[1]))
    de = np.zeros((dim_e, p.shape[1]))
    for i in np.arange(dim_e): 
        if not flexible:
            np.random.seed(70+i)
            canonic = np.random.multivariate_normal(np.zeros(points.shape[1]),covar)
            canonic = canonic/(2*np.max(np.abs(canonic))) # set it [-0.5,05]
            np.random.seed(env+i*50)
            if variance == -1:
                nu = np.random.uniform(-0.4,0.4) # [-0.9, 0.9]
            else:
                nu = np.random.normal(0, np.max([variance,0.4/2]))
            eofp = nu + canonic
        else:
            np.random.seed(env+i)
            if variance == -1:
                eofp = np.random.multivariate_normal(np.zeros(points.shape[1]),covar)
                eofp = 0.9* eofp/(np.max(np.abs(eofp))) # [-0.9,0.9]
            else:
                eofp = np.random.multivariate_normal(np.zeros(points.shape[1]),variance*covar)



        # upsample cause gram matrix is limiting
        factor = 50
        if dim_pcs == 1:
            eofp = np.repeat(eofp, factor)

            filter = np.ones(factor)/factor
            eofp = scipy.ndimage.convolve1d(eofp, filter, mode='nearest')

            step_up, points_up = mesh(factor*num_bins, dim_pcs)
        else:
            eofp = eofp.reshape((sqrtnum_bins,sqrtnum_bins))
            eofp = np.repeat(np.repeat(eofp, factor, axis=0), factor, axis=1)

            filter = np.ones((factor,factor))/(factor**2)

            t0 = time.time()
            eofp = scipy.signal.convolve2d(eofp, filter, mode='same', boundary='symm')
            t1 = time.time()
            print(t1-t0)
            
            step_up, points_up = mesh(factor**2*num_bins, dim_pcs)

        # Get path intersection
        # t0 = time.time()
        # eofplin = eofp.reshape(-1,)
        # for j in np.arange(eofplin.shape[0]):
        #     dist = np.max(np.abs(np.expand_dims(points_up[:,j],-1)-p), axis=0) # Manhattan distance
        #     de[i,np.argwhere(dist < step_up)] = eofplin[j]
        # t1 = time.time()
        # print(t1-t0)

        if dim_pcs == 1:
            t0 = time.time()
            origin = np.array([-1])[:,np.newaxis]
            idcs = (np.abs(p - origin) // (2*step_up)).astype(int)
            e[i,:] = eofp[idcs]
            t1 = time.time()
            print(t1-t0)

        else:
            t0 = time.time()
            origin = np.array([-1,1])[:,np.newaxis]
            idcs = (np.abs(p - origin) // (2*step_up)).astype(int)
            e[i,:] = eofp[idcs[0,:],idcs[1,:]]
            t1 = time.time()
            print(t1-t0)

        aux = np.diff(e[i,:])/dt
        de[i,:] = np.append(aux,aux[-1])
        print(np.max(de[i,:]))
        print(np.min(de[i,:]))
    return e, de, eofp