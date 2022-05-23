import numpy as np


class simple_traj:
    def __init__(self):
        self.info = "Simple trajectory planner"
        self.T = 5
        self.dt = .01
        self.N = 21

        self.index = 0

        self.t = np.linspace(0, self.T, int(self.T/self.dt))

        self.k = 3
        self.phi = np.sin(self.k*np.pi*self.t)
        self.theta = np.sin(self.k*np.pi*self.t)
        self.psi = np.sin(self.k*np.pi*self.t)
        self.p = self.k*np.pi*np.cos(self.k*np.pi*self.t)
        self.q = self.k*np.pi*np.cos(self.k*np.pi*self.t)
        self.r = self.k*np.pi*np.cos(self.k*np.pi*self.t)

    def sample(self):
        traj = np.zeros((6, self.N))
        self.index += 1
        for i in range(self.N):
            traj_t = np.array([self.phi[self.index+i],
                               self.theta[self.index+i],
                               self.psi[self.index+i],
                               self.p[self.index+i],
                               self.q[self.index+i],
                               self.r[self.index+i]])

            traj[:, i] = traj_t

        return traj

    def sample_nmpc(self):
        traj = np.zeros((6, self.N-1))
        self.index += 1
        for i in range(1, self.N):
            traj_t = np.array([self.phi[self.index+i],
                               self.theta[self.index+i],
                               self.psi[self.index+i],
                               self.p[self.index+i],
                               self.q[self.index+i],
                               self.r[self.index+i]])

            traj[:, i-1] = traj_t

        return traj

    def sample_cascate(self):
        traj = np.zeros((3, self.N))
        self.index += 1
        for i in range(self.N):
            traj_t = np.array([self.phi[self.index+i],
                               self.theta[self.index+i],
                               self.psi[self.index+i]])

            traj[:, i] = traj_t

        return traj

    def sample_cascate_nmpc(self):
        traj = np.zeros((3, self.N-1))
        self.index += 1
        for i in range(1, self.N):
            traj_t = np.array([self.phi[self.index+i],
                               self.theta[self.index+i],
                               self.psi[self.index+i]])

            traj[:, i-1] = traj_t

        return traj
