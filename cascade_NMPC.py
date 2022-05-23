from scipy import sparse
from cvxpy import Parameter, Variable, quad_form, Problem, Minimize, OSQP
import cvxpy as cp
import numpy as np
import casadi as ca


class CNMPCController:
    def __init__(self, optimizer='ipopt', A=np.zeros((3, 3)), B_rate=np.zeros((3, 3)), dt=0.01):
        self.info = "MPC attitude control for quadcopter"
        self.dt = dt
        self.optimizer = optimizer
        self.A = np.diag(np.ones(3))+A*self.dt
        self.B_rate = B_rate*self.dt
        [self.nx, self.nu] = self.B_rate.shape
        self.x_dim = self.nx
        self.u_dim = self.nu
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

        self._initDynamics()

    def _initDynamics(self,):
        phi, theta, psi = ca.MX.sym('phi'), ca.MX.sym('theta'), ca.MX.sym(
            'psi')

        self._x = ca.vertcat(phi, theta, psi)

        u1, u2, u3 = ca.MX.sym('u1'), ca.MX.sym('u2'), ca.MX.sym('u3')

        self._u = ca.vertcat(u1, u2, u3)

        x_dot = ca.vertcat(u1+ca.sin(phi)*ca.tan(theta)*u2+ca.cos(phi)*ca.tan(theta)*u3,
                           ca.cos(phi)*u2-ca.sin(phi)*u3,
                           u2*ca.sin(phi)/ca.cos(theta)+u3 *
                           ca.cos(phi)/ca.cos(theta))

        self.f = ca.Function('f', [self._x, self._u], [
                             x_dot], ['x', 'u'], ['ode'])

        intg_options = {'tf': .01, 'simplify': True,
                        'number_of_finite_elements': 4}
        dae = {'x': self._x, 'p': self._u, 'ode': self.f(self._x, self._u)}

        intg = ca.integrator('intg', 'rk', dae, intg_options)

        Delta_x = ca.SX.sym("Delta_x", self.x_dim)
        Delta_u = ca.SX.sym("Delta_u", self.u_dim)

        cost_track = Delta_x.T @ self.Q @ Delta_x
        cost_u = Delta_u.T @ self.R @ Delta_u

        f_cost_track = ca.Function('cost_track', [Delta_x], [cost_track])
        f_cost_u = ca.Function('cost_u', [Delta_u], [cost_u])

        x_next = intg(x0=self._x, p=self._u)['xf']
        F = ca.Function('F', [self._x, self._u], [
                        x_next], ['x', 'u'], ['x_next'])

        umin = self.umin_att.tolist()
        umax = self.umax_att.tolist()
        xmin = self.xmin_att.tolist()
        xmax = self.xmax_att.tolist()

        opti = ca.Opti()

        X = opti.variable(self.x_dim, self.N+1)
        self.P = opti.parameter(self.x_dim, self.N+1)
        self.U = opti.variable(self.u_dim, self.N)

        cost = 0

        opti.subject_to(X[:, 0] == self.P[:, 0])

        for k in range(self.N):
            cost += f_cost_track((X[:, k+1]-self.P[:, k+1]))
            if k == 0:
                cost += 0
            else:
                cost += f_cost_u((self.U[:, k]-self.U[:, k-1]))

            opti.subject_to(X[:, k+1] == F(X[:, k], self.U[:, k]))
            opti.subject_to(self.U[:, k] <= umax)
            opti.subject_to(self.U[:, k] >= umin)
            opti.subject_to(X[:, k+1] >= xmin)
            opti.subject_to(X[:, k+1] <= xmax)

        ipopt_options = {
            'verbose': False,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.max_iter": 100,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.print_level": 0,
            "print_time": False
        }
        """Slow
        ipopt_options = {
            'verbose': False,
            "ipopt.print_level": 0,
            "ipopt.warm_start_init_point": "yes",
            "print_time": False
        }"""
        qrqp_options = {'qpsol': 'qrqp',
                        'print_header': False,
                        'print_iteration': False,
                        'print_time': False,
                        'qpsol_options': {'print_iter': False,
                                          'print_header': False,
                                          'print_info': False}}
        opti.minimize(cost)
        if self.optimizer == 'ipopt':
            opti.solver('ipopt', ipopt_options)
        else:
            opti.solver('sqpmethod', qrqp_options)

        self.opti = opti

    def __update_objective_rate(self, x_init, xr):
        self.x_init_rate.value = x_init
        self.xr_rate.value = xr

    def run_controller(self, x, xr):
        p = np.concatenate((x[:3].reshape((3, 1)), xr), axis=1)

        self.opti.set_value(self.P, p)
        sol = self.opti.solve()
        u_att = sol.value(self.U)

        rate_traj = np.concatenate(
            (np.array(u_att), np.array(u_att)[:, self.N-1].reshape((3, 1))), axis=1).T.tolist()

        self.__update_objective_rate(x_init=x[-3:], xr=rate_traj)
        self.prob_rate.solve(solver=OSQP, warm_start=True, verbose=False)
        return self.u_rate[:, 0].value
