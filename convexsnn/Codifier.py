import numpy as np
class ProjectionCod():

    def codify(self, p, dp):
        
        time_steps = p.shape[1]
        x = np.zeros((p.shape[0] + 1,time_steps))
        dx = np.zeros((p.shape[0] + 1,time_steps))

        sqrt = np.sqrt(1 - np.linalg.norm(p, axis=0)**2)
        summ = np.sum(p * dp, axis = 0)

        x[:-1,:] = p
        x[-1,:] = sqrt

        dx[:-1,:] = dp
        dx[-1,:] = -summ / sqrt       

        return x, dx

    def decodify(self, x, dx):

        p = x[-1,:]
        dp = dx[-1,:]

        return p, dp

class DonutCod():

    def codify(self, p, dp, scale=1/2):
        
        time_steps = p.shape[1]
        x = np.zeros((3,time_steps))
        dx = np.zeros((3,time_steps))
        z = np.zeros((3,time_steps))
        dz = np.zeros((3,time_steps))

        # Define the module
        cycles = scale*np.pi

        p[0,:] = cycles * (p[0,:] + 1)
        p[1,:] = cycles * (p[1,:] + 1)
        dp[0,:] = cycles* (dp[0,:] + 1)
        dp[1,:] = cycles * (dp[1,:] + 1)

        x[0,:] = (1 + (1/2)*np.cos(p[0,:]))*np.cos(p[1,:])
        x[1,:] = (1 + (1/2)*np.cos(p[0,:]))*np.sin(p[1,:])
        x[2,:] = (1/2)*np.sin(p[0,:])

        dx[0,:] = (-(1/2)*np.sin(p[0,:])*dp[0,:])*np.cos(p[1,:]) - (1 + (1/2)*np.cos(p[0,:]))*np.sin(p[1,:])*dp[1,:]
        dx[1,:] = (-(1/2)*np.sin(p[0,:])*dp[0,:])*np.sin(p[1,:]) + (1 + (1/2)*np.cos(p[0,:]))*np.cos(p[1,:])*dp[1,:]
        dx[2,:] = (1/2)*np.cos(p[0,:])*dp[0,:]

        z[0,:] = np.cos(p[1,:])
        z[1,:] = np.sin(p[1,:])
        dz[0,:] = -np.sin(p[1,:])*dp[1,:]
        dz[1,:] = np.cos(p[1,:])*dp[1,:]

        return x, dx, z, dz

class TorusCod():

    def codify(self, p, dp, scale=1/2, type='square'):
        
        dim_pcs = p.shape[0]
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

        # Module define
        cycles = scale*np.pi
        alpha = np.zeros_like(p)
        dalpha = np.zeros_like(p)
        alpha[0,:] = cycles * (pr[0,:] + 1)
        if dim_pcs == 2: alpha[1,:] = cycles * (pr[1,:] + 1)
        dalpha[0,:] = cycles * dpr[0,:]
        if dim_pcs == 2: dalpha[1,:] = cycles * dpr[1,:]


        if type == 'square' or 'twisted' or 'rhombus':

            time_steps = p.shape[1]
            if dim_pcs == 2:
                x = np.zeros((4,time_steps))
                dx = np.zeros((4,time_steps))
            else:
                x = np.zeros((2,time_steps))
                dx = np.zeros((2,time_steps))

            twist = 1/2 if type == 'twisted' else 0

            x[0,:] = np.cos(alpha[0,:])
            x[1,:] = np.sin(alpha[0,:])
            if dim_pcs == 2:
                x[2,:] = np.cos(alpha[1,:] + twist*alpha[0,:])
                x[3,:] = np.sin(alpha[1,:] + twist*alpha[0,:])
        
            dx[0,:] = -np.sin(alpha[0,:])*dalpha[0,:]
            dx[1,:] = np.cos(alpha[0,:])*dalpha[0,:]
            if dim_pcs == 2:
                dx[2,:] = -np.sin(alpha[1,:] + twist*alpha[0,:])*(dalpha[1,:] + twist*dalpha[0,:])
                dx[3,:] = np.cos(alpha[1,:] + twist*alpha[0,:])*(dalpha[1,:] + twist*dalpha[0,:])

            # Normalize
            if dim_pcs == 2:
                x = 1/np.sqrt(2)*x
                dx = 1/np.sqrt(2)*dx

        elif type == '6D':

            time_steps = p.shape[1]
            x = np.zeros((6,time_steps))
            dx = np.zeros((6,time_steps))

            twist1 = 1/np.sqrt(3)
            twist2 = -1/np.sqrt(3)

            x[0,:] = np.cos(alpha[0,:])
            x[1,:] = np.sin(alpha[0,:])
            x[2,:] = np.cos(alpha[1,:] + twist1*alpha[0,:])
            x[3,:] = np.sin(alpha[1,:] + twist1*alpha[0,:])
            x[4,:] = np.cos(alpha[1,:] + twist2*alpha[0,:])
            x[5,:] = np.sin(alpha[1,:] + twist2*alpha[0,:])
        
            dx[0,:] = -np.sin(alpha[0,:])*dalpha[0,:]
            dx[1,:] = np.cos(alpha[0,:])*dalpha[0,:]
            dx[2,:] = -np.sin(alpha[1,:] + twist1*alpha[0,:])*(dalpha[1,:] + twist1*dalpha[0,:])
            dx[3,:] = np.cos(alpha[1,:] + twist1*alpha[0,:])*(dalpha[1,:] + twist1*dalpha[0,:])
            dx[4,:] = -np.sin(alpha[1,:] + twist2*alpha[0,:])*(dalpha[1,:] + twist2*dalpha[0,:])
            dx[5,:] = np.cos(alpha[1,:] + twist2*alpha[0,:])*(dalpha[1,:] + twist2*dalpha[0,:])

            # Normalize
            x = 1/np.sqrt(3)*x
            dx = 1/np.sqrt(3)*dx

        return x, dx

    def decodify(self, y, scale=1/2, type='square'):

        dim_pcs = y.shape[0]/2
        if scale > 1 or type != 'square':
            #TODO
            raise Exception("Decoding still not implemented")
        else:
            alpha = np.zeros((2,y.shape[1]))
            p_hat = np.zeros((2,y.shape[1]))

            alpha[0,:] = np.arctan2(y[1,:],y[0,:])
            alpha[0,alpha[0,:] < 0] += 2*np.pi
            if dim_pcs == 2:
                alpha[1,:] = np.arctan2(y[3,:],y[2,:])
                alpha[1,alpha[1,:] < 0] += 2*np.pi

            cycles = scale*np.pi
            p_hat[0,:] = alpha[0,:]/cycles - 1
            if dim_pcs == 2:
                p_hat[1,:] = alpha[1,:]/cycles - 1

        return p_hat
        
