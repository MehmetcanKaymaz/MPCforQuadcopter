from cProfile import label
import numpy as np
import matplotlib.pyplot as plt


class QuadModel:
    def __init__(self, model_type="complex", integrator="euler", n_iter=10):
        self.info = "3 DOF quadcopter dynamic model"
        self.model_type = model_type
        self.integrator = integrator
        self.x = np.zeros(6)
        self.x_dot = np.zeros(6)
        self.u = np.zeros(3)
        self.I = np.array([[1e-3, 0, 0],
                           [0, 1e-3, 0],
                           [0, 0, 1e-3]])
        self.dt = 1e-3
        self.n_iter = n_iter
        self.state_data = []
        self.u_data = []
        self.traj_data = []
        self.traj_err = 0

    def __update_x_dot(self):
        phi, theta, psi, p, q, r = self.x.tolist()
        self.x_dot = np.array([p+np.cos(phi)*np.tan(theta)*r+np.sin(phi)*np.tan(theta)*q,
                               np.cos(phi)*q-np.sin(phi)*r,
                               r*np.cos(phi)/np.cos(theta)+q *
                               np.sin(phi)/np.cos(theta),
                               (self.I[1, 1]-self.I[2, 2])*q*r /
                               self.I[0, 0]+self.u[0]/self.I[0, 0],
                               (self.I[2, 2]-self.I[0, 0])*p*r /
                               self.I[1, 1]+self.u[1]/self.I[1, 1],
                               (self.I[0, 0]-self.I[1, 1])*q*p /
                               self.I[2, 2]+self.u[2]/self.I[2, 2]])

    def __update_x(self):
        self.x += self.x_dot*self.dt
        self.state_data.append(self.x.copy())
        self.u_data.append(self.u.copy())

    def reset(self, x0=np.zeros(6)):
        self.x = x0
        self.u = np.zeros(3)
        self.x_dot = np.zeros(6)

        return self.x

    def step(self, u, traj):
        self.u = u
        self.traj_data.append(traj[:, 0])
        for i in range(self.n_iter):
            self.__update_x_dot()
            self.__update_x()
        traj = traj[:, 0].copy()
        err = 0
        for i in range(3):
            err += (traj[i]-self.x[i])**2

        self.traj_err += np.sqrt(err)

        return self.x

    def get_A_B(self):
        phi, theta, psi, p, q, r = self.x.tolist()
        A = np.array([[0, 0, 0, 1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
                      [0, 0, 0, 0, np.cos(phi), -np.sin(phi)],
                      [0, 0, 0, 0, np.sin(phi)/np.cos(theta),
                     np.cos(phi)/np.cos(theta)],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]])

        B = np.array([[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0],
                      [1/self.I[0, 0], 0, 0],
                      [0, 1/self.I[1, 1], 0],
                      [0, 0, 1/self.I[2, 2]]])

        return A, B

    def get_B_rate(self):
        return np.array([[1/self.I[0, 0], 0, 0],
                         [0, 1/self.I[1, 1], 0],
                         [0, 0, 1/self.I[2, 2]]])

    def get_B_att(self):
        phi, theta, psi, p, q, r = self.x.tolist()
        B_att = np.array([[1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
                          [0, np.cos(phi), -np.sin(phi)],
                          [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]])

        return B_att

    def get_all_data(self):
        return np.array(self.state_data), np.array(self.u_data), np.array(self.traj_data)

    def vis_result(self):
        state_name = ["phi", "theta", "psi", "p", "q", "r"]
        self.state_data = np.array(self.state_data)
        self.traj_data = np.array(self.traj_data)
        t1 = np.linspace(
            0, len(self.state_data[:, 0])*self.dt, len(self.state_data[:, 0]))
        t2 = np.linspace(
            0, len(self.traj_data[:, 0])*self.dt*10, len(self.traj_data[:, 0]))
        for i in range(len(state_name)):
            plt.subplot(3, 3, i+1)
            plt.plot(t1, self.state_data[:, i], label=state_name[i])
            if i < 3:
                plt.plot(t2, self.traj_data[:, i], label=state_name[i]+"_ref")
            plt.legend()
        u_names = ["u1", "u2", "u3"]
        self.u_data = np.array(self.u_data)
        for i in range(3):
            plt.subplot(3, 3, 6+i+1)
            plt.plot(t1, self.u_data[:, i], label=u_names[i])

        plt.show()
