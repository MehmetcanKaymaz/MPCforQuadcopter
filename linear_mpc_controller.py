from scipy import sparse
from cvxpy import Parameter, Variable, quad_form, Problem, Minimize, OSQP
import cvxpy as cp
import numpy as np


class LMPCController:
    def __init__(self, A=np.zeros((6, 6)), B=np.zeros((6, 3)), dt=0.01):
        self.info = "MPC attitude control for quadcopter"
        self.dt = dt
        self.A = np.diag(np.ones(6))+A*self.dt
        self.B = B*self.dt
        [self.nx, self.nu] = self.B.shape
        self.umin = np.array([-2, -2, -2])
        self.umax = np.array([2, 2, 2])
        self.xmax = np.array(
            [np.pi/2, np.pi/2, np.pi, 10*np.pi, 10*np.pi, 10*np.pi])
        self.xmin = -self.xmax

        self.Q = sparse.diags([5, 5, 5, .001, .001, .001])
        self.QN = sparse.diags([5, 5, 5, .01, .01, .01])
        self.R = 0.001*sparse.eye(3)

        self.N = 20

        self.u = Variable((self.nu, self.N))
        self.x = Variable((self.nx, self.N+1))
        self.x_init = Parameter(self.nx)
        self.Ad = Parameter((self.nx, self.nx), value=self.A)
        self.Bd = Parameter((self.nx, self.nu), value=self.B)
        self.objective = 0
        self.constraints = [self.x[:, 0] == self.x_init]
        self.xr = Parameter((self.nx, self.N+1))
        for k in range(self.N):
            self.objective += quad_form(self.x[:, k] - self.xr[:, k],
                                        self.Q) + quad_form(self.u[:, k], self.R)
            self.constraints += [self.x[:, k+1] == self.Ad @
                                 self.x[:, k] + self.Bd@self.u[:, k]]
            self.constraints += [self.xmin <= self.x[:, k],
                                 self.x[:, k] <= self.xmax]
            self.constraints += [self.umin <= self.u[:, k],
                                 self.u[:, k] <= self.umax]
        self.objective += quad_form(self.x[:,
                                    self.N] - self.xr[:, self.N], self.QN)
        self.prob = Problem(Minimize(self.objective), self.constraints)

    def __update_objective(self, x_init, xr):
        self.x_init.value = x_init
        self.xr.value = xr

    def run_controller(self, x, xr):
        self.__update_objective(x, xr)
        self.prob.solve(solver=OSQP, warm_start=True, verbose=False)

        return self.u[:, 0].value
