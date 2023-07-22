import numpy as np
class AngleEncoder():

    def encode(self, g, dg):
        
        dim = g.shape[0]
        
        # pe to alpha
        cycles = np.pi/2
        alpha = cycles * (g + 1)
        dalpha = cycles * dg

        # alpha to S1
        time_steps = g.shape[1]
        k = np.zeros((2*dim,time_steps))
        dk = np.zeros((2*dim,time_steps))
        
        j = 0
        for i in np.arange(dim):
            k[j,:] = np.cos(alpha[i,:])
            k[j+1,:] = np.sin(alpha[i,:])

            dk[j,:] = -np.sin(alpha[i,:])*dalpha[i,:]
            dk[j+1,:] = np.cos(alpha[i,:])*dalpha[i,:]
            j += 2
    
        # Normalize
        k = 1/np.sqrt(dim)*k
        dk = 1/np.sqrt(dim)*dk

        return k, dk

    def decode(self, s_hat):

        dim = int(s_hat.shape[0]/2)
        time_steps = s_hat.shape[1]
        alpha = np.zeros((dim,time_steps))

        # S1 to alpha
        j = 0
        for i in np.arange(dim):
            alpha[i,:] = np.arctan2(s_hat[j+1,:],s_hat[j,:])
            alpha[i,alpha[i,:] < -np.pi/2] += 2*np.pi
            j += 2

        # alpha to pe
        cycles = np.pi/2
        g_hat = alpha/cycles - 1

        return g_hat