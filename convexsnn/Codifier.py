import numpy as np

class ProjectionCod():

    def codify(self, p, dp, A, Theta):
        dbox = Theta.shape[0] + 1
        time_steps = p.shape[1]
        x = np.zeros((dbox,time_steps))
        dx = np.zeros((dbox,time_steps))

        sqrt = np.sqrt(1 - np.linalg.norm(p, axis=0)**2)
        summ = np.sum(p * dp, axis = 0)

        x[:-1,:] = Theta @ p
        x[-1,:] = A * sqrt

        dx[:-1,:] = Theta @ dp
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
        