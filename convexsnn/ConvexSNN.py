import numpy as np

class ConvexSNN():
    def __init__(self, lamb, F, Om, T):
        self.lamb = lamb
        self.F = F
        self.Om = Om
        self.T = T
        self.n = self.Om.shape[0]

    def simulate(self, c, I, V0=0, r0=0, dt=0.001, time_steps=100):
        V = np.zeros((self.n,time_steps))
        s = np.zeros((self.n,time_steps))
        r = np.zeros((self.n,time_steps))

        V[:,0] = V[:,0] + V0
        r[:,0] = r[:,0] + r0
        
        Vtp = V[:,0]
        stp = s[:,0]
        rtp = r[:,0]

        for i in range(time_steps-1):
            Vt = Vtp + self.Om @ stp
            rt = rtp + stp

            Vtp = Vt + dt*(-self.lamb*Vt + self.F @ c[:,i] + I[:,i])
            stp.fill(0)
            stp[Vtp > self.T] = 1
            rtp = rt + dt*(-self.lamb*rt)

            V[:,i+1] = Vtp
            s[:,i+1] = stp
            r[:,i+1] = rtp
        
        return V, s, r

    def simulate_adaptive(self, c, I, V0=0, r0=0, dt=0.001, time_steps=100, a=5):
        V = np.zeros((self.n,time_steps))
        s = np.zeros((self.n,time_steps))
        r = np.zeros((self.n,time_steps))

        V[:,0] = V[:,0] + V0
        r[:,0] = r[:,0] + r0
        
        Vtp = V[:,0]
        stp = s[:,0]
        rtp = r[:,0]

        aTtp = np.zeros_like(self.T)
        for i in range(time_steps-1):
            Vt = Vtp + self.Om @ stp
            rt = rtp + stp
            aTt = aTtp + a*stp

            Vtp = Vt + dt*(-self.lamb*Vt + self.F @ c[:,i] + I[:,i])
            stp.fill(0)
            stp[Vtp > aTtp + self.T] = 1
            aTtp = aTt + dt*(-self.lamb*aTt)
            rtp = rt + dt*(-self.lamb*rt)

            V[:,i+1] = Vtp
            s[:,i+1] = stp
            r[:,i+1] = rtp
        
        return V, s, r