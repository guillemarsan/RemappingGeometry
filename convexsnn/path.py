
import numpy as np

def get_path(dpcs, type, path_seed=0):
    dt = 0.0001
    time_steps = int(3 * 1/dt)
    t = np.arange(time_steps)*dt

    ones = np.ones((dpcs,time_steps))
    tmax = np.max(t)

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