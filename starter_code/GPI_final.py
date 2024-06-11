import numpy as np
from value_function import GridValueFunction
import utils
import tqdm
from dataclasses import dataclass
from discretization import create_state_space
from joblib import Parallel, delayed
import os
import argparse

class GPI:
    def __init__(self, config, nt, batch_size=10):
        self.config = config
        self.transition_matrix = None  # Initialize transition matrix variable
        self.stage_costs = None  # Initialize stage costs variable
        self.nt = nt  # Total number of time steps
        self.policy = np.zeros((nt, len(config.ex_space), len(config.ey_space), len(config.eth_space), 2))
        self.lr = 0.98  # Learning rate
        self.batch_size = batch_size  # Batch size for parallel processing

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

    @utils.timer
    def policy_evaluation(self):
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

        for iter in range(self.config.num_evals):
            new_value_function = self.config.V.copy()
            print(f"{iter}--Mean value: {np.mean(new_value_function.value_function)}")
            # Vectorize the outer loop over t
            for t in range(self.nt - 2, -1, -1):
                ex, ey, eth = np.meshgrid(
                    self.config.ex_space,
                    self.config.ey_space,
                    self.config.eth_space,
                    indexing='ij'
                )

                ex, ey, eth = ex.flatten(), ey.flatten(), eth.flatten()
                i, j, k = ex_indices.flatten(), ey_indices.flatten(), eth_indices.flatten()

                v, w = self.policy[t, i, j, k, 0], self.policy[t, i, j, k, 1]
                v_idx, w_idx = self.control_metric_to_index(np.vstack([v, w]).T.T)

                transition_indices = self.transition_matrix[t, i, j, k, v_idx, w_idx, :, :3].astype(int)
                transition_probs = self.transition_matrix[t, i, j, k, v_idx, w_idx, :, 3]

                # Vectorize the future value computation
                next_i, next_j, next_k = transition_indices[:, :, 0], transition_indices[:, :, 1], transition_indices[:, :, 2]
                future_values = self.config.V(t + 1, self.config.ex_space[next_i], self.config.ey_space[next_j], self.config.eth_space[next_k])

                # Correct the shape for einsum
                transition_probs = transition_probs.reshape(-1, transition_probs.shape[-1])
                future_values = future_values.reshape(-1, future_values.shape[-1])

                future_value = np.einsum('ij,ij->i', transition_probs, future_values).reshape(-1)

                cost = self.stage_costs[t, i, j, k, v_idx, w_idx, 0] + self.config.gamma * future_value
                cost = self.config.V(t, ex, ey, eth) + self.lr * (cost - self.config.V(t, ex, ey, eth))
                new_value_function.update(t, ex, ey, eth, cost)
                #if mean value function change is less than 1e-6, break

            if np.mean(np.abs(new_value_function.value_function - self.config.V.value_function)) < 1e-3:
                print(f"At Iter_{iter} Converged!")
                break
            self.config.V.copy_from(new_value_function)

    def improve_policy_batch(self, t, batch):
        results = []
        for ix, iy, it in batch:
            i, j, k = ix, iy, it

            # Vectorized future values computation
            transition_indices = self.transition_matrix[t, i, j, k, :, :, :, :3].astype(int)
            transition_probs = self.transition_matrix[t, i, j, k, :, :, :, 3]

            next_ex = self.config.ex_space[transition_indices[:, :, :, 0]]
            next_ey = self.config.ey_space[transition_indices[:, :, :, 1]]
            next_eth = self.config.eth_space[transition_indices[:, :, :, 2]]

            # Compute the future values using the current policy
            future_values_flat = self.config.V(t + 1, next_ex, next_ey, next_eth)
            future_values = np.einsum('ijk,ijk->ij', transition_probs, future_values_flat)

            cost = self.stage_costs[t, i, j, k, :, :, 0] + self.config.gamma * future_values
            min_cost_idx = np.unravel_index(np.argmin(cost), cost.shape)
            results.append((t, ix, iy, it, self.control_index_to_metric(min_cost_idx[0], min_cost_idx[1])))
        return results

    @utils.timer
    def policy_improvement(self):
        """
        Policy improvement step of the GPI algorithm.
        """
        print("Start policy improvement")
        ex_indices, ey_indices, eth_indices = np.meshgrid(
            np.arange(len(self.config.ex_space)),
            np.arange(len(self.config.ey_space)),
            np.arange(len(self.config.eth_space)),
            indexing='ij'
        )

        time_batches = [list(range(self.nt - 1))[i:i + self.batch_size] for i in range(0, self.nt - 1, self.batch_size)]

        for time_batch in tqdm.tqdm(time_batches, desc="Policy Improvement"):
            results = Parallel(n_jobs=-1)(
                delayed(self.improve_policy_batch)(t, list(zip(ex_indices.flatten(), ey_indices.flatten(), eth_indices.flatten())))
                for t in time_batch
            )

            for result_batch in results:
                for t, ix, iy, it, control in result_batch:
                    self.policy[t, ix, iy, it, :] = control

    def compute_policy(self, num_iters: int) -> None:
        print("Start computing policy")
        num_checkpoint = 2
        for i in range(num_iters):
            print(f"Iteration {i}")
            self.policy_evaluation()
            self.policy_improvement()
            
            if i % num_checkpoint == 0:
                self.save_policy(i)

        self.save_policy(num_iters)

    def save_policy(self, num_iters: int) -> None:
        import os
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)
        np.savez(self.config.output_dir + f'/policy_{num_iters}', policy=self.policy)

    def load_policy(self, filename: str) -> None:
        print("loading policy...")
        with np.load(filename) as data:
            self.policy = data['policy']
            print("policy loaded!") 

    def policy_control(self, t: int, cur_state: np.ndarray) -> np.ndarray:
        i, j, k = self.state_metric_to_index(cur_state)
        return self.policy[t, i, j, k, :]

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='GPI_pp4_1')
    exp = parser.parse_args().exp

    ex_space, ey_space, eth_space, v_space, w_space = create_state_space()

    obstacles = np.array([[1.0, 2.0], [-2.0, -2.0]])
    Q = np.eye(2)  # State cost matrix
    q = 1.0  # Scalar cost for theta error
    R = np.eye(2)  # Control cost matrix
    gamma = 0.90  # Discount factor
    num_evals = 100  # Number of policy evaluations in each iteration
    collision_margin = 0.55  # Collision margin
    output_dir = f'./output/{exp}'  # Output directory for saving results

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
        V=None,  # Will be initialized in the GPI class
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

    data_folder = f'./data/{exp}/'
    gpi = GPI(config, nt, batch_size=100)  # Adjust the batch size as needed
    gpi.load_transition_matrix(f'{data_folder}combined_transition_matrix_float16.npz')
    gpi.load_stage_costs(f'{data_folder}combined_stage_cost_matrix.npz')
    gpi.init_value_function()
    gpi.compute_policy(num_iters=300)
