import numpy as np
from value_function import GridValueFunction
import utils
import tqdm
from dataclasses import dataclass
from discretization import adaptive_grid_theta, adaptive_grid_xy, adaptive_control_grid_v, adaptive_control_grid_w

class GPI:
    def __init__(self, config, nt):
        self.config = config
        self.transition_matrix = None  # Initialize transition matrix variable
        self.stage_costs = None  # Initialize stage costs variable
        self.nt = nt  # Total number of time steps
        self.policy = np.zeros((nt, len(config.ex_space), len(config.ey_space), len(config.eth_space), 2))
        self.lr = 0.5  # Learning rate

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
            self.transition_matrix = data['arr_0'].astype(np.float32)

    def load_stage_costs(self, filename: str):
        """
        Read the stage costs from a file.
        Args:
            filename (str): The filename of the stage costs.
        """
        with np.load(filename) as data:
            self.stage_costs = data['arr_0'].astype(np.float32)

    def init_value_function(self):
        """
        Initialize the value function.
        """
        self.config.V = GridValueFunction(self.nt, self.config.ex_space, self.config.ey_space, self.config.eth_space)

    def evaluate_value_function(self,):
        """
        Evaluate the value function. Implement this function if you are using a feature-based value function.
        """
        print("start evaluating value function")
        time_bar = tqdm.tqdm(range(self.config.num_evals))
        for _ in time_bar:
            new_value_function = self.config.V.copy()
            for t in range(self.nt - 2, -1, -1):
                for ix, ex in enumerate(self.config.ex_space):
                    for iy, ey in enumerate(self.config.ey_space):
                        for it, eth in enumerate(self.config.eth_space):
                            i, j, k = self.state_metric_to_index(np.array([ex, ey, eth]))
                            v, w = self.policy[t, i, j, k, :]
                            v_idx, w_idx = self.control_metric_to_index(np.array([v, w]))
                            transition_indices = self.transition_matrix[t, i, j, k, v_idx, w_idx, :, :3].astype(int)
                            transition_probs = self.transition_matrix[t, i, j, k, v_idx, w_idx, :, 3]   
                            future_value = 0

                            for (next_i, next_j, next_k), prob in zip(transition_indices, transition_probs):
                                future_value += prob * self.config.V(t + 1, self.config.ex_space[next_i], self.config.ey_space[next_j], self.config.eth_space[next_k])
                            cost = self.stage_costs[t, i, j, k, v_idx, w_idx, 0] + self.config.gamma * future_value
                            cost = self.config.V(t, ex, ey, eth) + self.lr * (cost - self.config.V(t, ex, ey, eth))
                            new_value_function.update(t, ex, ey, eth, cost)

            self.config.V.copy_from(new_value_function)

    @utils.timer
    def policy_improvement(self):
        """
        Policy improvement step of the GPI algorithm.
        """
        print("start policy improvement")
        for t in range(self.nt-1):
            for ix, ex in enumerate(self.config.ex_space):
                for iy, ey in enumerate(self.config.ey_space):
                    for it, eth in enumerate(self.config.eth_space):
                        i, j, k = self.state_metric_to_index(np.array([ex, ey, eth]))
                        future_values = np.zeros((len(self.config.v_space), len(self.config.w_space)))
                        for iv, v in enumerate(self.config.v_space):
                            for iw, w in enumerate(self.config.w_space):
                                transition_indices = self.transition_matrix[t, i, j, k, iv, iw, :, :3].astype(int)
                                transition_probs = self.transition_matrix[t, i, j, k, iv, iw, :, 3]
                                future_value = 0
                                for (next_i, next_j, next_k), prob in zip(transition_indices, transition_probs):
                                    future_value += prob * self.config.V(t + 1, self.config.ex_space[next_i], self.config.ey_space[next_j], self.config.eth_space[next_k])
                                future_values[iv, iw] = future_value
                        cost = self.stage_costs[t, i, j, k, :, :, 0] + self.config.gamma * future_values
                        min_cost_idx = np.unravel_index(np.argmin(cost), cost.shape)
                        self.policy[t, i, j, k, :] = self.control_index_to_metric(min_cost_idx[0], min_cost_idx[1])

    @utils.timer
    def policy_evaluation(self):
        """
        Policy evaluation step of the GPI algorithm.
        """
        self.evaluate_value_function()

    def compute_policy(self, num_iters: int) -> None:

        print("start computing policy")
        """
        Compute the policy for a given number of iterations.
        Args:
            num_iters (int): number of iterations
        """
        for i in range(num_iters):
            print(f"iteration {i}")
            self.policy_evaluation()
            self.policy_improvement()

        #save policy
        #if not exit output directory, create it
        import os
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)
        np.savez(self.config.output_dir + '/policy', policy=self.policy)

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


    @utils.timer
    def policy_evaluation0(self):
        """
        Policy evaluation step of the GPI algorithm.
        """
        print("Start evaluating value function")

        ex_indices, ey_indices, eth_indices = np.meshgrid(
            np.arange(len(self.config.ex_space)),
            np.arange(len(self.config.ey_space)),
            np.arange(len(self.config.eth_space)),
            indexing='ij'
        )

        for _ in range(self.config.num_evals):
            new_value_function = self.config.V.copy()
            for t in range(self.nt - 2, -1, -1):
                for ix, iy, it in zip(ex_indices.flatten(), ey_indices.flatten(), eth_indices.flatten()):
                    ex, ey, eth = self.config.ex_space[ix], self.config.ey_space[iy], self.config.eth_space[it]
                    i, j, k = ix, iy, it

                    v, w = self.policy[t, i, j, k, :]
                    v_idx, w_idx = self.control_metric_to_index(np.array([v, w]))
                    transition_indices = self.transition_matrix[t, i, j, k, v_idx, w_idx, :, :3].astype(int)
                    transition_probs = self.transition_matrix[t, i, j, k, v_idx, w_idx, :, 3]

                    # Vectorize the future value computation
                    future_values = self.config.V(t + 1, self.config.ex_space[transition_indices[:, 0]],
                                                self.config.ey_space[transition_indices[:, 1]],
                                                self.config.eth_space[transition_indices[:, 2]])
                    future_value = np.dot(transition_probs, future_values)

                    cost = self.stage_costs[t, i, j, k, v_idx, w_idx, 0] + self.config.gamma * future_value
                    cost = self.config.V(t, ex, ey, eth) + self.lr * (cost - self.config.V(t, ex, ey, eth))
                    new_value_function.update(t, ex, ey, eth, cost)

            self.config.V.copy_from(new_value_function)

    @utils.timer
    def policy_evaluation_(self):
        """
        Policy evaluation step of the GPI algorithm.
        """
        print("Start evaluating value function")
        # time_bar = tqdm.tqdm(range(self.config.num_evals))
        for _ in range(self.config.num_evals):
            new_value_function = self.config.V.copy()
            for t in range(self.nt - 2, -1, -1):
                ex_indices, ey_indices, eth_indices = np.meshgrid(
                    np.arange(len(self.config.ex_space)),
                    np.arange(len(self.config.ey_space)),
                    np.arange(len(self.config.eth_space)),
                    indexing='ij'
                )
                for ix, iy, it in zip(ex_indices.flatten(), ey_indices.flatten(), eth_indices.flatten()):
                    ex, ey, eth = self.config.ex_space[ix], self.config.ey_space[iy], self.config.eth_space[it]
                    i, j, k = self.state_metric_to_index(np.array([ex, ey, eth]))

                    v, w = self.policy[t, i, j, k, :]
                    v_idx, w_idx = self.control_metric_to_index(np.array([v, w]))
                    transition_indices = self.transition_matrix[t, i, j, k, v_idx, w_idx, :, :3].astype(int)
                    transition_probs = self.transition_matrix[t, i, j, k, v_idx, w_idx, :, 3]

                    # Vectorize the future value computation
                    future_values = self.config.V(t + 1, self.config.ex_space[transition_indices[:, 0]],
                                                  self.config.ey_space[transition_indices[:, 1]],
                                                  self.config.eth_space[transition_indices[:, 2]])
                    future_value = np.dot(transition_probs, future_values)

                    cost = self.stage_costs[t, i, j, k, v_idx, w_idx, 0] + self.config.gamma * future_value
                    cost = self.config.V(t, ex, ey, eth) + self.lr * (cost - self.config.V(t, ex, ey, eth))
                    new_value_function.update(t, ex, ey, eth, cost)

            self.config.V.copy_from(new_value_function)
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
    V: GridValueFunction  # your value function implementation
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

if __name__ == "__main__":
    nt = 100  # Replace with the actual number of time steps

    traj = utils.lissajous
    # ex_space = adaptive_grid_xy()    
    # ey_space = adaptive_grid_xy()
    # eth_space = adaptive_grid_theta()
    # v_space = adaptive_control_grid_v()
    # w_space = adaptive_control_grid_w()

    #toy space for testing
    ex_space = np.linspace(-2, 2, 3)
    ey_space = np.linspace(-2, 2, 3)
    eth_space = np.linspace(-np.pi, np.pi, 3)
    v_space = np.linspace(0, 1, 3)
    w_space = np.linspace(-1, 1, 3)


    obstacles = np.array([[1.0, 2.0], [-2.0, -2.0]])
    Q = np.eye(2)  # State cost matrix
    q = 1.0  # Scalar cost for theta error
    R = np.eye(2)  # Control cost matrix
    gamma = 0.99  # Discount factor
    num_evals = 1  # Number of policy evaluations in each iteration
    collision_margin = 0.55  # Collision margin
    output_dir = './output'  # Output directory for saving results

    config = GpiConfig(
        traj=traj,
        obstacles=obstacles,
        ex_space=ex_space,
        ey_space=ey_space,
        eth_space=eth_space,
        v_space=v_space,
        w_space=w_space,
        Q=Q,
        q=q,
        R=R,
        gamma=gamma,
        num_evals=num_evals,
        collision_margin=collision_margin,
        V=None,   # Will be initialized in the GPI class
        output_dir=output_dir,
        v_ex_space=None,
        v_ey_space=None,
        v_etheta_space=None,
        v_alpha=None,
        v_beta_t=None,
        v_beta_e=None,
        v_lr=None,
        v_batch_size=None
    )
    data_folder = './data/toy/' 
    gpi = GPI(config, nt)
    gpi.load_transition_matrix(f'{data_folder}combined_transition_matrix_float16_toy.npz')
    gpi.load_stage_costs(f'{data_folder}combined_stage_cost_matrix_toy.npz')
    gpi.init_value_function()
    gpi.compute_policy(num_iters=1)
