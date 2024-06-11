from dataclasses import dataclass
import numpy as np
from value_function import ValueFunction
import utils
from starter_code.trans.transition_matrix import compute_transition_matrix
from starter_code.trans.stage_cost import compute_stage_cost_matrix
from value_function import GridValueFunction
@dataclass
class GpiConfig:
    traj: callable
    obstacles: np.ndarray
    ex_space: np.ndarray
    ey_space: np.ndarray
    eth_space: np.ndarray
    v_space: np.ndarray
    w_space: np.ndarray
    Q: np.ndarray
    q: float
    R: np.ndarray
    gamma: float
    num_evals: int  # number of policy evaluations in each iteration
    collision_margin: float
    V: ValueFunction  # your value function implementation
    output_dir: str
    # used by feature-based value function
    v_ex_space: np.ndarray
    v_ey_space: np.ndarray
    v_etheta_space: np.ndarray
    v_alpha: float
    v_beta_t: float
    v_beta_e: float
    v_lr: float
    v_batch_size: int  # batch size if GPU memory is not enough

class GPI:
    def __init__(self, config: GpiConfig):
        self.config = config
        self.transition_matrix = None  # Initialize transition matrix variable
        self.stage_costs = None  # Initialize stage costs variable
        self.policy = np.zeros((self.config.num_evals, len(self.config.ex_space), len(self.config.ey_space), len(self.config.eth_space), 2))
        
    def state_metric_to_index(self, metric_state: np.ndarray) -> tuple:
        ex, ey, eth = metric_state
        i = np.digitize(ex, self.config.ex_space, right=True) - 1
        j = np.digitize(ey, self.config.ey_space, right=True) - 1
        k = np.digitize(eth, self.config.eth_space, right=True) - 1
        return i, j, k

    def state_index_to_metric(self, state_index: tuple) -> np.ndarray:
        i, j, k = state_index
        ex = self.config.ex_space[i]
        ey = self.config.ey_space[j]
        eth = self.config.eth_space[k]
        return np.array([ex, ey, eth])
    
    def control_metric_to_index(self, control_metric: np.ndarray) -> tuple:
        v = np.digitize(control_metric[0], self.config.v_space, right=True)
        w = np.digitize(control_metric[1], self.config.w_space, right=True)
        return v, w

    def control_index_to_metric(self, v: np.ndarray, w: np.ndarray) -> tuple:
        return self.config.v_space[v], self.config.w_space[w]

    def load_transition_matrix(self, filename: str):
        """
        Read the transition matrix from a file.
        Args:
            filename (str): The filename of the transition matrix.
        """
        with np.load(filename) as data:
            self.transition_matrix = data['transition_matrix_float16'].astype(np.float32)

    def load_stage_costs(self, filename: str):
        """
        Read the stage costs from a file.
        Args:
            filename (str): The filename of the stage costs.
        """
        with np.load(filename) as data:
            self.stage_costs = data['stage_costs_float16'].astype(np.float32)
    
    def init_value_function(self):
        """
        Initialize the value function.
        """
        self.config.V = GridValueFunction(self.config.num_evals, self.config.ex_space, self.config.ey_space, self.config.eth_space)

    def evaluate_value_function(self):
        """
        Evaluate the value function. Implement this function if you are using a feature-based value function.
        """
        for _ in range(self.config.num_evals):
            new_value_function = self.config.V.copy()
            for t in range(self.config.num_evals):
                for ix, ex in enumerate(self.config.ex_space):
                    for iy, ey in enumerate(self.config.ey_space):
                        for it, eth in enumerate(self.config.eth_space):
                            i, j, k = self.state_metric_to_index(np.array([ex, ey, eth]))
                            cost = self.stage_costs[t, i, j, k, :, :, 0] + self.config.gamma * np.dot(self.transition_matrix[t, i, j, k, :, :, 3], self.config.V(t+1, ex, ey, eth))
                            new_value_function.update(t, ex, ey, eth, np.min(cost))
            self.config.V.copy_from(new_value_function)

    @utils.timer
    def policy_improvement(self):
        """
        Policy improvement step of the GPI algorithm.
        """
        for t in range(self.config.num_evals):
            for ix, ex in enumerate(self.config.ex_space):
                for iy, ey in enumerate(self.config.ey_space):
                    for it, eth in enumerate(self.config.eth_space):
                        i, j, k = self.state_metric_to_index(np.array([ex, ey, eth]))
                        cost = self.stage_costs[t, i, j, k, :, :, 0] + self.config.gamma * np.dot(self.transition_matrix[t, i, j, k, :, :, 3], self.config.V(t+1, ex, ey, eth))
                        min_cost_idx = np.unravel_index(np.argmin(cost), cost.shape)
                        self.policy[t, i, j, k, :] = self.control_index_to_metric(min_cost_idx[0], min_cost_idx[1])

    @utils.timer
    def policy_evaluation(self):
        """
        Policy evaluation step of the GPI algorithm.
        """
        self.evaluate_value_function()

    def compute_policy(self, num_iters: int) -> None:
        """
        Compute the policy for a given number of iterations.
        Args:
            num_iters (int): number of iterations
        """
        for _ in range(num_iters):
            self.policy_improvement()
            self.policy_evaluation()

    def __call__(self, t: int, cur_state: np.ndarray, cur_ref_state: np.ndarray) -> np.ndarray:
        """
        Given the time step, current state, and reference state, return the control input.
        Args:
            t (int): time step
            cur_state (np.ndarray): current state
            cur_ref_state (np.ndarray): reference state
        Returns:
            np.ndarray: control input
        """
        i, j, k = self.state_metric_to_index(cur_state)
        return self.policy[t, i, j, k, :]


if __name__ == "__main__":
    config = GpiConfig(
        traj=...,
        obstacles=...,
        ex_space=...,
        ey_space=...,
        eth_space=...,
        v_space=...,
        w_space=...,
        Q=...,
        q=...,
        R=...,
        gamma=...,
        num_evals=...,
        collision_margin=...,
        V=...,
        output_dir=...,
        v_ex_space=...,
        v_ey_space=...,
        v_etheta_space=...,
        v_alpha=...,
        v_beta_t=...,
        v_beta_e=...,
        v_lr=...,
        v_batch_size=...
    )

    gpi = GPI(config)
    gpi.load_transition_matrix('transition_matrix_float16.npz')
    gpi.load_stage_costs('stage_cost_matrix.npz')
    gpi.init_value_function()
    gpi.compute_policy(num_iters=10)