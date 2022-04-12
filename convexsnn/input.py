
import numpy as np

def get_input(inp, type, amp, dir=1):
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
        x = A*ones
        dx = ones*0

    elif type == 'circle' or type == 'dis-circle':
        sp = 5
        time_steps = int(2*np.pi/(dt*sp))
        t = np.arange(time_steps)*dt

        if type == 'dis-circle':
            pts = 8
            step = int(time_steps/pts)
            deg = np.repeat(t[np.arange(time_steps,step=step)],step)
            time_steps = deg.shape[0]
            t = np.arange(time_steps)*dt
        else:
            deg = t

        ones = np.ones((inp,time_steps))
        x = np.cos(sp*deg)*A*ones
        if inp > 1: x[1,:] = np.sin(sp*deg)*A[1]*ones[1,:]

        if type == 'dis-circle':
            dx = ones*0
        else:
            dx = -A*sp*np.sin(sp*deg)*ones
            if inp > 1: dx[1,:] = A[1]*sp*np.cos(sp*deg)*ones[1,:]

        

    elif type == 'spiral' or type == 'dis-spiral':
        osp = 0.05
        csp = 3
        time_steps = int(np.pi/(2*(dt*osp)))
        t = np.arange(time_steps)*dt
        
        if type == 'dis-spiral':
            pts = 100
            step = int(time_steps/pts)
            deg = np.repeat(t[np.arange(time_steps,step=step)],step)
            time_steps = deg.shape[0]
            t = np.arange(time_steps)*dt
        else:
            deg = t

        tmax = np.max(t)
        nint = (tmax-deg)/tmax
        ones = np.ones((inp,time_steps))

        x = np.cos(csp*deg)*nint*A*ones
        if inp > 1: x[1,:] = np.sin(csp*deg)*nint*A[1]*ones[1,:]
        if inp > 2: x[2,:] = np.sin(osp*deg)*A[0]*ones[2,:]

        if type == 'dis-spiral':
            dx = ones*0
        else:
            dx = -(1/tmax)*A*np.cos(csp*deg)*ones - nint*A*csp*np.sin(csp*deg)
            if inp > 1: dx[1,:] = -(1/tmax)*A[1]*np.sin(csp*deg)*ones[1,:] + nint*A[1]*csp*np.cos(csp*deg)*ones[1,:]
            if inp > 2: dx[2,:] = A[0]*osp*np.cos(osp*deg)*ones[2,:]

    elif type == 'semicircle':
        sp = 5
        dir_vect = np.ones(inp-1)*dir/np.linalg.norm(dir)
        time_steps = int(np.pi/(dt*sp))
        t = np.arange(time_steps)*dt
        ones = np.ones((inp,time_steps))
    
        x = np.copy(ones)
        if inp > 1:
            x[:-1,:] = A[0]*dir_vect[:,None]*np.cos(sp*t)*ones[:-1,:]
            x[-1,:] = A[0]*np.sin(sp*t)*ones[-1,:]

        dx = 0*ones
        if inp > 1:
            dx[:-1,:] = -A[0]*dir_vect[:,None]*sp*np.sin(sp*t)*ones[:-1,:]
            dx[-1,:] = A[0]*sp*np.cos(sp*t)*ones[-1,:]

    return x, dx, t, dt, time_steps