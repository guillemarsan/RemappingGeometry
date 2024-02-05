import numpy as np
import cvxpy as cp
from utils import check_code_encode

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
        stp = np.zeros(self.n)

        V[:,0] = V[:,0] + V0
        r[:,0] = r[:,0] + r0
        
        for i in range(1,time_steps):
            stp.fill(0)
            stp[np.where(V[:,i-1]>self.T)] = 1
            V[:,i] = V[:,i-1] + dt*(-self.lamb*V[:,i-1] + self.F @ c[:,i] + I[:,i-1]) + self.Om @ stp
            r[:,i] = r[:,i-1] + dt*(-self.lamb*r[:,i-1]) + stp
            s[:,i] = np.copy(stp)
        
        return V, s, r
    def simulate_one(self, c, I, V0=0, r0=0, dt=0.001, time_steps=100):
        V = np.zeros((self.n,time_steps))
        s = np.zeros((self.n,time_steps))
        r = np.zeros((self.n,time_steps))
        stp = np.zeros(self.n)

        V[:,0] = V[:,0] + V0
        r[:,0] = r[:,0] + r0
        
        for i in range(1,time_steps):
            stp.fill(0)
            if any(V[:,i-1]>self.T):
                stp[np.random.choice(np.where(V[:,i-1]>self.T)[0])] = 1
            V[:,i] = V[:,i-1] + dt*(-self.lamb*V[:,i-1] + self.F @ c[:,i] + I[:,i-1]) + self.Om @ stp
            r[:,i] = r[:,i-1] + dt*(-self.lamb*r[:,i-1]) + stp
            s[:,i] = np.copy(stp)
        
        return V, s, r

    def simulate_pathint(self, dx, I, decoder, x0=0, V0=0, r0=0, dt=0.001, time_steps=100):
        V = np.zeros((self.n,time_steps))
        s = np.zeros((self.n,time_steps))
        r = np.zeros((self.n,time_steps))
        x = np.zeros((dx.shape[0],time_steps))
        stp = np.zeros(self.n)

        V[:,0] = V[:,0] + V0
        r[:,0] = r[:,0] + r0
        x[:,0] = x[:,0] + x0
        
        for i in range(1,time_steps):
            stp.fill(0)
            stp[np.where(V[:,i-1]>self.T)] = 1
            # V[:,i] = V[:,i-1] + dt*(-self.lamb*V[:,i-1] + self.F @ (self.lamb*x[:,i-1] + dx[:,i]) + I[:,i-1]) + self.Om @ stp
            V[:,i] = V[:,i-1] + dt*(self.F @ (self.lamb*x[:,i-1] + dx[:,i]) + I[:,i-1]) + self.Om @ stp
            r[:,i] = r[:,i-1] + dt*(-self.lamb*r[:,i-1]) + stp
            x[:,i] = decoder(r[:,i], i)
            s[:,i] = np.copy(stp)

        return V, s, r, x

    def simulate_pathint_one(self, dx, I, decoder, x0=0, V0=0, r0=0, dt=0.001, time_steps=100):
        V = np.zeros((self.n,time_steps))
        s = np.zeros((self.n,time_steps))
        r = np.zeros((self.n,time_steps))
        x = np.zeros((dx.shape[0],time_steps))
        stp = np.zeros(self.n)

        V[:,0] = V[:,0] + V0
        r[:,0] = r[:,0] + r0
        x[:,0] = x[:,0] + x0
        
        for i in range(1,time_steps):
            stp.fill(0)
            if any(V[:,i-1]>self.T):
                stp[np.random.choice(np.where(V[:,i-1]>self.T)[0])] = 1
            V[:,i] = V[:,i-1] + dt*(self.F @ (self.lamb*x[:,i-1] + dx[:,i]) + I[:,i-1]) + self.Om @ stp
            # r[:,i] = r[:,i-1] + dt*(-1.037*self.lamb*r[:,i-1]) + stp
            r[:,i] = r[:,i-1] + dt*(-self.lamb*r[:,i-1]) + stp
            x[:,i] = decoder(r[:,i], i)
            # x[:,i] = x[:,i-1] + dt*dx[:,i]
            s[:,i] = np.copy(stp)

        return V, s, r, x

            # Slomo implementation
            # C = np.sum(Vtp > self.T)
            # while C > 0:
            #     w = np.argmax(Vtp-self.T)
            #     stp[w] = 1
            #     Vtp = Vtp + self.Om[w,:]
            #     rtp[w] = rtp[w] + 1 
            #     C = np.sum(Vtp > self.T)

    def simulate_minimization(self, x):
        
        np.random.seed(0)
        xp = cp.Parameter(x.shape[0])
        r = cp.Variable(self.n)
        objective = cp.Minimize(cp.sum_squares(xp - self.F.T @ r) + 2*r.T @ self.T)
        constraints = [r >= 0]
        prob = cp.Problem(objective, constraints)

        rt = []
        for xval in x.T:
            xp.value = xval
            l = prob.solve(solver=cp.ECOS) #feastol
            rt.append(r.value)
        rt = np.vstack(rt).T
        print(np.min(rt))
        rt[rt < 0] = 0 #non feasible results
        # check_code_encode(x, rt, self.F.T, self.T)
        return rt

        # Batch attempt: not converging
        # np.random.seed(0)
        # r = cp.Variable((self.n, x.shape[1]))
        # objective = cp.Minimize(cp.sum(cp.sum_squares(x - self.F.T @ r) + r.T @ self.T))
        # constraints = [r >= 0]
        # prob = cp.Problem(objective, constraints)
        # l = prob.solve(solver=cp.SCS, eps=1e-8, verbose=True) #ECOS, feastol
        # return r

    def simulate_adaptive(self, c, I, V0=0, r0=0, dt=0.001, time_steps=100, a=5):
        V = np.zeros((self.n,time_steps))
        s = np.zeros((self.n,time_steps))
        r = np.zeros((self.n,time_steps))
        stp = np.zeros(self.n)
        atp = np.zeros(self.n)

        V[:,0] = V[:,0] + V0
        r[:,0] = r[:,0] + r0
        
        for i in range(1,time_steps):
            stp.fill(0)
            if any(V[:,i-1]>self.T + atp[:,i-1]):
                stp[np.random.choice(np.where(V[:,i-1]>self.T)[0])] = 1
            V[:,i] = V[:,i-1] + dt*(-self.lamb*V[:,i-1] + self.F @ c[:,i] + I[:,i-1]) + self.Om @ stp
            r[:,i] = r[:,i-1] + dt*(-self.lamb*r[:,i-1]) + stp
            atp[:,i] = atp[:,i-1] + dt*(-self.lamb*atp[:,i-1]) + a*stp
            s[:,i] = np.copy(stp)
        
        return V, s, r