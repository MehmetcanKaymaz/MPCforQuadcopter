import casadi as ca
import numpy as np
from scipy import sparse


class NMPC:
    def __init__(self, optimizer='ipopt'):
        self.info = "Nonlinear Model Predictive Control for Quadcopter Attitudes"
        self.optimizer = optimizer
        self._dt = 1e-2
        self._N = 20

        self._x_dim = 6
        self._u_dim = 3

        self._I = np.diag([1e-3, 1e-3, 1e-3])

        self._Q = sparse.diags([5, 5, 5, .001, .001, .001])
        self._R = 0.001*sparse.eye(3)

        self._x0 = np.zeros(6)
        self._u0 = np.zeros(3)

        self._initDynamics()

    def _initDynamics(self,):
        phi, theta, psi, p, q, r = ca.MX.sym('phi'), ca.MX.sym('theta'), ca.MX.sym(
            'psi'), ca.MX.sym('p'), ca.MX.sym('q'), ca.MX.sym('r')

        self._x = ca.vertcat(phi, theta, psi, p, q, r)

        u1, u2, u3 = ca.MX.sym('u1'), ca.MX.sym('u2'), ca.MX.sym('u3')

        self._u = ca.vertcat(u1, u2, u3)

        x_dot = ca.vertcat(p+ca.sin(phi)*ca.tan(theta)*q+ca.cos(phi)*ca.tan(theta)*r,
                           ca.cos(phi)*q-ca.sin(phi)*r,
                           q*ca.sin(phi)/ca.cos(theta)+r *
                           ca.cos(phi)/ca.cos(theta),
                           u1/self._I[0, 0],
                           u2/self._I[1, 1],
                           u3/self._I[2, 2])

        self.f = ca.Function('f', [self._x, self._u], [
                             x_dot], ['x', 'u'], ['ode'])

        intg_options = {'tf': .01, 'simplify': True,
                        'number_of_finite_elements': 4}
        dae = {'x': self._x, 'p': self._u, 'ode': self.f(self._x, self._u)}

        intg = ca.integrator('intg', 'rk', dae, intg_options)

        Delta_x = ca.SX.sym("Delta_x", self._x_dim)
        Delta_u = ca.SX.sym("Delta_u", self._u_dim)

        cost_track = Delta_x.T @ self._Q @ Delta_x
        cost_u = Delta_u.T @ self._R @ Delta_u

        f_cost_track = ca.Function('cost_track', [Delta_x], [cost_track])
        f_cost_u = ca.Function('cost_u', [Delta_u], [cost_u])

        x_next = intg(x0=self._x, p=self._u)['xf']
        F = ca.Function('F', [self._x, self._u], [
                        x_next], ['x', 'u'], ['x_next'])

        umin = [-2, -2, -2]
        umax = [2, 2, 2]
        xmin = [-np.pi/2, -np.pi/2, -np.pi, -10*np.pi, -10*np.pi, -10*np.pi]
        xmax = [np.pi/2, np.pi/2, np.pi, 10*np.pi, 10*np.pi, 10*np.pi]

        opti = ca.Opti()

        X = opti.variable(self._x_dim, self._N+1)
        self.P = opti.parameter(self._x_dim, self._N+1)
        self.U = opti.variable(self._u_dim, self._N)

        cost = 0

        opti.subject_to(X[:, 0] == self.P[:, 0])

        for k in range(self._N):
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

    def run_controller(self, x0, ref_states):

        p = np.concatenate((x0.reshape((6, 1)), ref_states), axis=1)

        self.opti.set_value(self.P, p)
        sol = self.opti.solve()
        u = sol.value(self.U)[:, 0]
        return u
