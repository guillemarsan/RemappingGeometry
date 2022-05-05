import numpy as np
import matplotlib.pyplot as plt
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

class TorusCod():

    def codify(self, p, dp, A, Theta):
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
        ax = plt.axes(projection='3d')
        #ax.plot3D(x[0,:], x[1,:],x[2,:])

        dx[0,:] = (-(A/2)*np.sin(p[0,:])*dp[0,:])*np.cos(p[1,:]) - (A + (A/2)*np.cos(p[0,:]))*np.sin(p[1,:])*dp[1,:]
        dx[1,:] = (-(A/2)*np.sin(p[0,:])*dp[0,:])*np.sin(p[1,:]) + (A + (A/2)*np.cos(p[0,:]))*np.cos(p[1,:])*dp[1,:]
        dx[2,:] = (A/2)*np.cos(p[0,:])*dp[0,:]

        z[0,:] = A*np.cos(p[1,:])
        z[1,:] = A*np.sin(p[1,:])
        #ax.plot3D(z[0,:], z[1,:], z[2,:])
        dz[0,:] = -A*np.sin(p[1,:])*dp[1,:]
        dz[1,:] = A*np.cos(p[1,:])*dp[1,:]

        return x, dx, z, dz

class Torus4DCod():

    def codify(self, p, dp, A, Theta):
        dbox = Theta.shape[0] + 1
        time_steps = p.shape[1]
        x = np.zeros((dbox,time_steps))
        dx = np.zeros((dbox,time_steps))

        vect1 = np.array([np.sqrt(3)/4,1/4])
        vect2 = np.array([1/4,np.sqrt(3)/4])

        pr = np.zeros_like(p)
        dpr = np.zeros_like(p)
        pr[0,:] = vect1[0]*p[0,:] + vect2[0]*p[1,:]
        pr[1,:] = vect1[1]*p[0,:] + vect2[1]*p[1,:]
        dpr[0,:] = vect1[0]*dp[0,:] + vect2[0]*dp[1,:]
        dpr[1,:] = vect1[1]*dp[0,:] + vect2[1]*dp[1,:]

        cycles = 4*np.pi

        prc = np.zeros_like(p)
        dprc = np.zeros_like(p)
        prc[0,:] = cycles * (pr[0,:] + 1)
        prc[1,:] = cycles * (pr[1,:] + 1)
        dprc[0,:] = cycles * (dpr[0,:] + 1)
        dprc[1,:] = cycles * (dpr[1,:] + 1)

        r = A/np.sqrt(2)

        x[0,:] = r*np.cos(prc[1,:])
        x[1,:] = r*np.sin(prc[1,:])
        x[2,:] = r*np.cos(prc[0,:])
        x[3,:] = r*np.sin(prc[0,:])
    
        dx[0,:] = -r*np.sin(prc[1,:])*dprc[1,:]
        dx[1,:] = r*np.cos(prc[1,:])*dprc[1,:]
        dx[2,:] = -r*np.sin(prc[0,:])*dprc[0,:]
        dx[3,:] = r*np.cos(prc[0,:])*dprc[0,:]

        return x, dx
        
