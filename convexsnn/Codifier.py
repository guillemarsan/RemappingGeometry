import numpy as np
class ProjectionCod():

    def codify(self, p, dp, A, Theta):
        dbox = Theta.shape[0] + 1
        time_steps = p.shape[1]
        x = np.zeros((dbox,time_steps))
        dx = np.zeros((dbox,time_steps))

        sqrt = np.sqrt(1 - np.linalg.norm(p, axis=0)**2)
        summ = np.sum(p * dp, axis = 0)

        x[:-1,:] = A * Theta @ p
        x[-1,:] = A * sqrt

        dx[:-1,:] = A * Theta @ dp
        dx[-1,:] = -A * summ / sqrt       

        return x, dx

    def decodify(self, x, dx, A, Theta):
        dpcs = Theta.shape[0]
        time_steps = p.shape[1]
        p = np.zeros((dpcs,time_steps))
        dp = np.zeros((dpcs,time_steps))

        p = 1/A * Theta.T @ x[-1,:]
        dp = 1/A * Theta.T @ dx[-1,:]

        return p, dp

class DonutCod():

    def codify(self, p, dp, A, Theta, type='square'):
        dbox = Theta.shape[0] + 1
        time_steps = p.shape[1]
        x = np.zeros((dbox,time_steps))
        dx = np.zeros((dbox,time_steps))
        z = np.zeros((dbox,time_steps))
        dz = np.zeros((dbox,time_steps))

        p[0,:] = np.pi/2 * (p[0,:] + 1)
        p[1,:] = np.pi/3 * (p[1,:] + 1)
        dp[0,:] = np.pi/2 * (dp[0,:] + 1)
        dp[1,:] = np.pi/3 * (dp[1,:] + 1)

        x[0,:] = (A + (A/2)*np.cos(p[0,:]))*np.cos(p[1,:])
        x[1,:] = (A + (A/2)*np.cos(p[0,:]))*np.sin(p[1,:])
        x[2,:] = (A/2)*np.sin(p[0,:])

        dx[0,:] = (-(A/2)*np.sin(p[0,:])*dp[0,:])*np.cos(p[1,:]) - (A + (A/2)*np.cos(p[0,:]))*np.sin(p[1,:])*dp[1,:]
        dx[1,:] = (-(A/2)*np.sin(p[0,:])*dp[0,:])*np.sin(p[1,:]) + (A + (A/2)*np.cos(p[0,:]))*np.cos(p[1,:])*dp[1,:]
        dx[2,:] = (A/2)*np.cos(p[0,:])*dp[0,:]

        z[0,:] = A*np.cos(p[1,:])
        z[1,:] = A*np.sin(p[1,:])
        dz[0,:] = -A*np.sin(p[1,:])*dp[1,:]
        dz[1,:] = A*np.cos(p[1,:])*dp[1,:]

        return x, dx, z, dz

class TorusCod():

    def codify(self, p, dp, A, Theta, type='square'):
        
        dbox = Theta.shape[0] + 1
        time_steps = p.shape[1]
        x = np.zeros((dbox,time_steps))
        dx = np.zeros((dbox,time_steps))

        if type == 'rhombus':
            vect1 = np.array([1,0])
            vect2 = np.array([1/2,np.sqrt(3)/2])
    
            pr = np.zeros_like(p)
            dpr = np.zeros_like(p)
            pr[0,:] = vect1[0]*p[0,:] + vect2[0]*p[1,:]
            pr[1,:] = vect1[1]*p[0,:] + vect2[1]*p[1,:]
            dpr[0,:] = vect1[0]*dp[0,:] + vect2[0]*dp[1,:]
            dpr[1,:] = vect1[1]*dp[0,:] + vect2[1]*dp[1,:]
        else:
            pr = p.copy()
            dpr = dp.copy()

        # Scale
        # TODO Incorporate A (r) and cycles to define scale (a module)
        cycles = 4*np.pi
        r = A/np.sqrt(2)

        alpha = np.zeros_like(p)
        dalpha = np.zeros_like(p)
        alpha[0,:] = cycles * (pr[0,:] + 1)
        alpha[1,:] = cycles * (pr[1,:] + 1)
        dalpha[0,:] = cycles * (dpr[0,:] + 1)
        dalpha[1,:] = cycles * (dpr[1,:] + 1)

        
        if type == 'square' or 'twisted' or 'rhombus':
            twist = 1/2 if type == 'twisted' else 0

            x[0,:] = r*np.cos(alpha[0,:])
            x[1,:] = r*np.sin(alpha[0,:])
            x[2,:] = r*np.cos(alpha[1,:] + twist*alpha[0,:])
            x[3,:] = r*np.sin(alpha[1,:] + twist*alpha[0,:])
        
            dx[0,:] = -r*np.sin(alpha[0,:])*dalpha[0,:]
            dx[1,:] = r*np.cos(alpha[0,:])*dalpha[0,:]
            dx[2,:] = -r*np.sin(alpha[1,:] + twist*alpha[0,:])*(dalpha[1,:] + twist*dalpha[0,:])
            dx[3,:] = r*np.cos(alpha[1,:] + twist*alpha[0,:])*(dalpha[1,:] + twist*dalpha[0,:])

        elif type == '6D':

            twist1 = 1/np.sqrt(3)
            twist2 = -1/np.sqrt(3)

            x[0,:] = r*np.cos(alpha[0,:])
            x[1,:] = r*np.sin(alpha[0,:])
            x[2,:] = r*np.cos(alpha[1,:] + twist1*alpha[0,:])
            x[3,:] = r*np.sin(alpha[1,:] + twist1*alpha[0,:])
            x[4,:] = r*np.cos(alpha[1,:] + twist2*alpha[0,:])
            x[5,:] = r*np.sin(alpha[1,:] + twist2*alpha[0,:])
        
            dx[0,:] = -r*np.sin(alpha[0,:])*dalpha[0,:]
            dx[1,:] = r*np.cos(alpha[0,:])*dalpha[0,:]
            dx[2,:] = -r*np.sin(alpha[1,:] + twist1*alpha[0,:])*(dalpha[1,:] + twist1*dalpha[0,:])
            dx[3,:] = r*np.cos(alpha[1,:] + twist1*alpha[0,:])*(dalpha[1,:] + twist1*dalpha[0,:])
            dx[4,:] = -r*np.sin(alpha[1,:] + twist2*alpha[0,:])*(dalpha[1,:] + twist2*dalpha[0,:])
            dx[5,:] = r*np.cos(alpha[1,:] + twist2*alpha[0,:])*(dalpha[1,:] + twist2*dalpha[0,:])


        return x, dx
        
