from scipy import sparse
from cvxpy import Parameter, Variable, quad_form, Problem, Minimize, OSQP
import cvxpy as cp
import numpy as np


class CMPCController:
    def __init__(self, A=np.zeros((3, 3)), B_att=np.zeros((3, 3)), B_rate=np.zeros((3, 3)), dt=0.01):
        self.info = "MPC attitude control for quadcopter"
        self.dt = dt
        self.A = np.diag(np.ones(3))+A*self.dt
        self.B_att = B_att*self.dt
        self.B_rate = B_rate*self.dt
        [self.nx, self.nu] = self.B_att.shape
        self.umin_rate = np.array([-2, -2, -2])
        self.umax_rate = np.array([2, 2, 2])
        self.umin_att = -np.array([10*np.pi, 10*np.pi, 10*np.pi])
        self.umax_att = np.array([10*np.pi, 10*np.pi, 10*np.pi])
        self.xmax_att = np.array(
            [np.pi/2, np.pi/2, np.pi])
        self.xmin_att = -self.xmax_att
        self.xmax_rate = np.array([10*np.pi, 10*np.pi, 10*np.pi])
        self.xmin_rate = -self.xmax_rate

        self.Q = sparse.diags([5, 5, 5])
        self.QN = sparse.diags([5, 5, 5])
        self.R = 0.001*sparse.eye(3)

        self.N = 20

        self.u_att = Variable((self.nu, self.N))
        self.x_att = Variable((self.nx, self.N+1))
        self.x_init_att = Parameter(self.nx)
        self.Ad_att = Parameter((self.nx, self.nx), value=self.A)
        self.Bd_att = Parameter((self.nx, self.nu), value=self.B_att)
        self.objective_att = 0
        self.constraints_att = [self.x_att[:, 0] == self.x_init_att]
        self.xr_att = Parameter((self.nx, self.N+1))
        for k in range(self.N):
            self.objective_att += quad_form(self.x_att[:, k] - self.xr_att[:, k],
                                            self.Q) + quad_form(self.u_att[:, k], self.R)
            self.constraints_att += [self.x_att[:, k+1] == self.Ad_att @
                                     self.x_att[:, k] + self.Bd_att@self.u_att[:, k]]
            self.constraints_att += [self.xmin_att <= self.x_att[:, k],
                                     self.x_att[:, k] <= self.xmax_att]
            self.constraints_att += [self.umin_att <= self.u_att[:, k],
                                     self.u_att[:, k] <= self.umax_att]
        self.objective_att += quad_form(self.x_att[:,
                                                   self.N] - self.xr_att[:, self.N], self.QN)
        self.prob_att = Problem(
            Minimize(self.objective_att), self.constraints_att)

        self.u_rate = Variable((self.nu, self.N))
        self.x_rate = Variable((self.nx, self.N+1))
        self.x_init_rate = Parameter(self.nx)
        self.Ad_rate = Parameter((self.nx, self.nx), value=self.A)
        self.Bd_rate = Parameter((self.nx, self.nu), value=self.B_rate)
        self.objective_rate = 0
        self.constraints_rate = [self.x_rate[:, 0] == self.x_init_rate]
        self.xr_rate = Parameter((self.nx, self.N+1))
        for k in range(self.N):
            self.objective_rate += quad_form(self.x_rate[:, k] - self.xr_rate[:, k],
                                             self.Q) + quad_form(self.u_rate[:, k], self.R)
            self.constraints_rate += [self.x_rate[:, k+1] == self.Ad_rate @
                                      self.x_rate[:, k] + self.Bd_rate@self.u_rate[:, k]]
            self.constraints_rate += [self.xmin_rate <= self.x_rate[:, k],
                                      self.x_rate[:, k] <= self.xmax_rate]
            self.constraints_rate += [self.umin_rate <= self.u_rate[:, k],
                                      self.u_rate[:, k] <= self.umax_rate]
        self.objective_rate += quad_form(self.x_rate[:,
                                                     self.N] - self.xr_rate[:, self.N], self.QN)
        self.prob_rate = Problem(
            Minimize(self.objective_rate), self.constraints_rate)

    def __update_objective_att(self, x_init, xr):
        self.x_init_att.value = x_init
        self.xr_att.value = xr

    def __update_objective_rate(self, x_init, xr):
        self.x_init_rate.value = x_init
        self.xr_rate.value = xr

    def __update_B_att(self, B):
        self.Bd_att.value = B*self.dt

    def run_controller(self, B_att, x, xr):
        self.__update_B_att(B=B_att)
        self.__update_objective_att(x[:3], xr)
        self.prob_att.solve(solver=OSQP, warm_start=True, verbose=False)
        rate_traj = np.concatenate(
            (np.array(self.u_att.value), np.array(self.u_att.value)[:, self.N-1].reshape((3, 1))), axis=1).T.tolist()

        self.__update_objective_rate(x_init=x[-3:], xr=rate_traj)
        self.prob_rate.solve(solver=OSQP, warm_start=True, verbose=False)
        return self.u_rate[:, 0].value
