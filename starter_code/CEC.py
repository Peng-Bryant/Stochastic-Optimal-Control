import casadi as ca
import numpy as np
import utils

class CEC:
    def __init__(self, config):
        self.config = config
        # Define optimization horizon
        self.T = config['T']
        # Define cost matrices
        self.Q = config['Q']
        self.R = config['R']
        self.q = config['q']
        # Define control bounds
        self.v_min = config['v_min']
        self.v_max = config['v_max']
        self.w_min = config['w_min']
        self.w_max = config['w_max']
        self.gamma = config['gamma']
        # Obstacles
        self.obstacles = config['obstacles']
        # Define environment bounds
        self.env_min = np.array([-3, -3])
        self.env_max = np.array([3, 3])

    def __call__(self, t: int, cur_state: np.ndarray) -> np.ndarray:
        """
        Given the time step and current state, return the control input.
        Args:
            t (int): time step
            cur_state (np.ndarray): current state
        Returns:
            np.ndarray: control input
        """
        # Define optimization variables
        U = ca.SX.sym('U', 2 * self.T)  # [v_0, w_0, ..., v_{T-1}, w_{T-1}]  # 20
        X = ca.SX.sym('X', 3 * (self.T + 1))  # [x_0, y_0, theta_0, ..., x_T, y_T, theta_T] # 33

        # Initial state
        X0 = cur_state

        # Define the cost function
        cost = 0
        constraints = []
        lbg = []
        ubg = []

        for i in range(self.T):
            # State at time i
            x_i = X[3 * i: 3 * (i + 1)]
            # Control at time i
            u_i = U[2 * i: 2 * (i + 1)]
            # Reference state at time i
            ref_i = utils.lissajous(t + i)

            # Compute error
            error = x_i - ref_i
            error[2] = ca.atan2(ca.sin(error[2]), ca.cos(error[2]))

            # Add to cost
            p_error = error[:2]
            cost += (ca.mtimes([p_error.T, self.Q, p_error]) + self.q * (1 - ca.cos(error[2]))**2 + ca.mtimes([u_i.T, self.R, u_i])) * self.gamma**i

            # Dynamics constraints: motion model
            next_x = X[3 * (i + 1): 3 * (i + 2)]
            f = ca.vertcat(
                x_i[0] + self.config['dt'] * (u_i[0] * ca.cos(x_i[2])),
                x_i[1] + self.config['dt'] * (u_i[0] * ca.sin(x_i[2])),
                x_i[2] + self.config['dt'] * u_i[1]
            )
            constraints.append(next_x - f)
            lbg.extend([0, 0, 0])
            ubg.extend([0, 0, 0])

            # Free space constraints
            constraints.append(x_i[:2] - self.env_min)  # Ensure within environment lower bounds
            lbg.extend([0, 0])  # Lower bound for free space constraint
            ubg.extend([ca.inf, ca.inf])  # Upper bound for free space constraint

            constraints.append(self.env_max - x_i[:2])  # Ensure within environment upper bounds
            lbg.extend([0, 0])  # Lower bound for free space constraint
            ubg.extend([ca.inf, ca.inf])  # Upper bound for free space constraint


            # Obstacle avoidance constraints
            for obs in self.obstacles:
                obs_center = obs[:2]
                obs_radius = obs[2]
                constraints.append(ca.norm_2(x_i[:2] - obs_center) - obs_radius)
                lbg.append(0.1)
                ubg.append(ca.inf)

        # Initial state constraint
        constraints.append(X[:3] - X0)
        lbg.extend([0, 0, 0])
        ubg.extend([0, 0, 0])

        # Control input constraints
        lbx = np.concatenate([np.tile([self.v_min, self.w_min], self.T), np.full(3 * (self.T + 1), -ca.inf)])
        ubx = np.concatenate([np.tile([self.v_max, self.w_max], self.T), np.full(3 * (self.T + 1), ca.inf)])
        # import pdb; pdb.set_trace()
        # Define the NLP problem
        nlp = {
            'x': ca.vertcat(U, X),
            'f': cost,
            'g': ca.vertcat(*constraints)
        }

        # Define the solver
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # Solve the NLP problem
        sol = solver(
            x0=np.zeros(2 * self.T + 3 * (self.T + 1)),  # Initial guess
            lbx=lbx,
            ubx=ubx,
            lbg=np.array(lbg),
            ubg=np.array(ubg)
        )

        # Extract the control input from the solution
        U_opt = sol['x'][:2 * self.T].full().flatten()
        control_input = U_opt[:2]
        return control_input

# Example usage
config = {
    'T': 10,
    'Q': np.diag([1, 1]),
    'R': np.diag([0.1, 0.1]),
    'q': 1.0,
    'v_min': 0,
    'v_max': 1,
    'w_min': -1,
    'w_max': 1,
    'dt': 0.5,
    'gamma': 0.9,
    'obstacles': np.array([[-2, -2, 0.5], [1, 2, 0.5]])
}

cec_controller = CEC(config)
