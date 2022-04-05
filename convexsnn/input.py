
import numpy as np

def get_input(inp, type, amp):
    dt = 0.0001
    time_steps = 1000
    t = np.arange(time_steps)*dt
    ones = np.ones((inp,time_steps))

    A = np.ones((inp,1))*amp
    if type == 'sine':
        sp = 100
        if inp > 1: A[1] *= -1

        x = np.sin(sp*t)*A*ones
        dx = sp*np.cos(sp*t)*A*ones
    elif type == 'lin':
        if inp > 1: A[1] *= -1
        x = A*t*ones
        dx = A*ones

    elif type == 'cst':
        if inp > 1: A[1] *= -1
        x = 0*ones
        x[2,:] = A[0]*ones[2,:]
        dx = ones*0

    elif type == 'circle':
        sp = 5
        time_steps = int(2*np.pi/(dt*sp))
        t = np.arange(time_steps)*dt
        ones = np.ones((inp,time_steps))

        x = np.cos(sp*t)*A*ones
        if inp > 1: x[1,:] = np.sin(sp*t)*A[1]*ones[1,:]
        dx = -A*sp*np.sin(sp*t)*ones
        if inp > 1: dx[1,:] = A[1]*sp*np.cos(sp*t)*ones[1,:]

    elif type == 'spiral':
        osp = 0.05
        csp = 3
        time_steps = int(np.pi/(2*(dt*osp)))
        t = np.arange(time_steps)*dt
        tmax = np.max(t)
        nint = (tmax-t)/tmax
        ones = np.ones((inp,time_steps))


        x = np.cos(csp*t)*nint*A*ones
        if inp > 1: x[1,:] = np.sin(csp*t)*nint*A[1]*ones[1,:]
        if inp > 2: x[2,:] = np.sin(osp*t)*A[0]*ones[2,:]
        dx = -(1/tmax)*A*np.cos(csp*t)*ones - nint*A*csp*np.sin(csp*t)
        if inp > 1: dx[1,:] = -(1/tmax)*A[1]*np.sin(csp*t)*ones[1,:] + A[1]*csp*np.cos(csp*t)*ones[1,:]
        if inp > 2: dx[2,:] = A[0]*osp*np.cos(osp*t)*ones[2,:]

    return x, dx, t, dt, time_steps