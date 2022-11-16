
import numpy as np

def get_path(dpcs, type):
    dt = 0.0001
    time_steps = 30000
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
        a = 20 # half circles

        p = nint*np.cos(a*t)*ones
        p[1,:] = nint*np.sin(a*t)*ones[1,:]
        dp = (-r0/tmax*np.cos(a*t) - nint*a*np.sin(a*t))*ones
        dp[1,:] = (-r0/tmax*np.sin(a*t) + nint*a*np.cos(a*t))*ones[1,:]

    
    return p, dp, t, dt, time_steps