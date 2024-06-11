import numpy as np
from joblib import Parallel, delayed
import tqdm
import utils
import time
from scipy.special import logsumexp
from discretization import adaptive_grid_theta, adaptive_grid_xy, adaptive_control_grid_v, adaptive_control_grid_w  


def mean_next_state(e, u, t, delta_t):
    r_x_curr, r_y_curr, r_theta_curr = lissajous_curve[t]
    r_x_next, r_y_next, r_theta_next = lissajous_curve[t + 1]

    G_e = np.array([[delta_t * np.cos(e[2] + r_theta_curr), 0],
                    [delta_t * np.sin(e[2] + r_theta_curr), 0],
                    [0, delta_t]])
    
    next_e = e + G_e @ u + np.array([r_x_curr - r_x_next, r_y_curr - r_y_next, r_theta_curr - r_theta_next])
    return next_e

def compute_stage_cost_for_state_control(tasks, delta_t, Q, q, R, k_col, collision_margin, ex_space, ey_space, eth_space, v_space, w_space, obstacles):
    results = []
    for task in tasks:
        t, ix, iy, it, iv, iw = task
        ex = ex_space[ix]
        ey = ey_space[iy]
        eth = eth_space[it]
        v = v_space[iv]
        w = w_space[iw]

        e = np.array([ex, ey, eth])
        u = np.array([v, w])
        next_e = mean_next_state(e, u, t, delta_t)

        # Compute stage cost
        p = e[:2]  # p is the position component of the error state
        pTQp = p.T @ Q @ p
        cost = pTQp + q * (1 - np.cos(e[2]))**2 + u.T @ R @ u

        p_next = next_e[:2]
        # Collision cost
        collision_cost = 0
        for obs in obstacles:
            distances = np.array([np.linalg.norm(p + lissajous_curve[t][:2] - obs), np.linalg.norm(p_next + lissajous_curve[t + 1][:2] - obs)])
            # if distance < collision_margin:
            #     collision_cost = k_col
            #     break
            collision_cost += np.sum(np.maximum(0, distances - collision_margin))

        total_cost = cost + collision_cost
        results.append((t, ix, iy, it, iv, iw, total_cost))
    return results

def compute_stage_cost_matrix(stage_cost_matrix, ex_space, ey_space, eth_space, v_space, w_space, delta_t, Q, q, R, k_col, collision_margin, obstacles, batch_size=100):
    tasks = []
    for t in range(nt):
        for ix in range(len(ex_space)):
            for iy in range(len(ey_space)):
                for it in range(len(eth_space)):
                    for iv in range(len(v_space)):
                        for iw in range(len(w_space)):
                            tasks.append((t, ix, iy, it, iv, iw))
    
    # Create batches of tasks
    task_batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
    
    start_time = time.time()
    results = Parallel(n_jobs=-1)(delayed(compute_stage_cost_for_state_control)(
        batch, delta_t, Q, q, R, k_col, collision_margin, ex_space, ey_space, eth_space, v_space, w_space, obstacles) for batch in tqdm.tqdm(task_batches))
    end_time = time.time()
    print(f"Time taken for parallel computation: {end_time - start_time:.2f} seconds")
    
    start_time = time.time()
    for result_batch in results:
        for t, ix, iy, it, iv, iw, total_cost in result_batch:
            stage_cost_matrix[t, ix, iy, it, iv, iw, 0] = total_cost
    
    end_time = time.time()
    print(f"Time taken for combining results: {end_time - start_time:.2f} seconds")
    import pdb; pdb.set_trace()
    # Save the stage cost matrix to a npz file
    np.savez_compressed('stage_cost_matrix.npz', stage_cost_matrix)


if __name__ == "__main__":
    # Example grid definitions
    error_threshold = 0.5
    fine_spacing = 0.1
    coarse_spacing = 0.6

    nt = 100
    ex_space = adaptive_grid_xy()    
    ey_space = adaptive_grid_xy()
    eth_space = adaptive_grid_theta()
    v_space = adaptive_control_grid_v()
    w_space = adaptive_control_grid_w()

    # Initialize transition matrix and stage cost matrix
    stage_cost_matrix = np.zeros((nt, len(ex_space), len(ey_space), len(eth_space), 
                                len(v_space), len(w_space), 1), dtype=np.float32)

    print("stage_cost_matrix shape:", stage_cost_matrix.shape)

    # Precompute the lissajous curve
    lissajous_curve = np.array([utils.lissajous(t) for t in range(101)])
    # Example usage
    Q = np.eye(2)  # Example value for Q
    q = 1.0  # Example value for q
    R = np.eye(2)  # Example value for R
    k_col = 1000.0  # Example value for collision cost
    collision_margin = 0.5 + 0.05  # Example value for collision margin
    obstacles = np.array([[1.0, 1.0], [-1.0, -1.0]])  # Example obstacles
    delta_t = 0.5

    compute_stage_cost_matrix(stage_cost_matrix, ex_space, ey_space, eth_space, v_space, w_space, delta_t, Q, q, R, k_col, collision_margin, obstacles, batch_size=2100)
