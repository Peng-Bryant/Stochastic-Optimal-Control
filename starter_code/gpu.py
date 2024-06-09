import numpy as np
from joblib import Parallel, delayed
import tqdm
import utils
import time

# Example grid definitions
error_threshold = 0.5
fine_spacing = 0.1
coarse_spacing = 0.6

nt = 1
ex_space = np.linspace(-3, 3, 20)
ey_space = np.linspace(-3, 3, 20)
eth_space = np.linspace(-np.pi, np.pi, 20)
v_space = np.linspace(0, 1, 10)
w_space = np.linspace(-1, 1, 10)

# Initialize transition matrix
transition_matrix = np.zeros((nt, len(ex_space), len(ey_space), len(eth_space), 
                              len(v_space), len(w_space), 8, 4), dtype=np.float32)

def mean_next_state(e, u, t, delta_t):
    r_x_curr, r_y_curr, r_theta_curr = utils.lissajous(t)
    r_x_next, r_y_next, r_theta_next = utils.lissajous(t + 1)

    G_e = np.array([[delta_t * np.cos(e[2] + r_theta_curr), 0],
                    [delta_t * np.sin(e[2] + r_theta_curr), 0],
                    [0, delta_t]])
    
    next_e = e + G_e @ u + np.array([r_x_curr - r_x_next, r_y_curr - r_y_next, r_theta_curr - r_theta_next])
    return next_e

def find_8_neighbors(mean_state, ex_space, ey_space, eth_space):
    i = np.digitize(mean_state[0], ex_space) - 1
    j = np.digitize(mean_state[1], ey_space) - 1
    k = np.digitize(mean_state[2], eth_space) - 1

    neighbors = []
    for di, dj, dk in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
        ni, nj, nk = i + di, j + dj, k + dk
        if 0 <= ni < len(ex_space) and 0 <= nj < len(ey_space) and 0 <= nk < len(eth_space):
            neighbors.append((ni, nj, nk))

    additional_points = [
        (i + 1, j + 1, k), (i - 1, j - 1, k),
        (i + 1, j, k + 1), (i - 1, j, k - 1),
        (i, j + 1, k + 1), (i, j - 1, k - 1)
    ]
    
    for ni, nj, nk in additional_points:
        if len(neighbors) >= 8:
            break
        if 0 <= ni < len(ex_space) and 0 <= nj < len(ey_space) and 0 <= nk < len(eth_space):
            neighbors.append((ni, nj, nk))

    return neighbors[:8]

def compute_transition_matrix_for_state_control(t, ix, ex, iy, ey, it, eth, iv, v, iw, w, delta_t, sigma, ex_space, ey_space, eth_space, v_space, w_space):
    local_transition_matrix = np.zeros((8, 4), dtype=np.float32)
    e = np.array([ex, ey, eth])
    u = np.array([v, w])

    mean_next_e = mean_next_state(e, u, t, delta_t)

    neighbors = find_8_neighbors(mean_next_e, ex_space, ey_space, eth_space)

    diff_matrix = np.array([[ex_space[ni], ey_space[nj], eth_space[nk]] for ni, nj, nk in neighbors]) - mean_next_e
    prob_matrix = np.exp(-0.5 * np.sum((diff_matrix / sigma) ** 2, axis=1))

    for idx, (ni, nj, nk) in enumerate(neighbors):
        local_transition_matrix[idx, 0] = ni
        local_transition_matrix[idx, 1] = nj
        local_transition_matrix[idx, 2] = nk
        local_transition_matrix[idx, 3] = prob_matrix[idx]

    return t, ix, iy, it, iv, iw, local_transition_matrix

def compute_transition_matrix(transition_matrix, ex_space, ey_space, eth_space, v_space, w_space, delta_t, sigma):
    tasks = []
    for t in range(nt):
        for ix, ex in enumerate(ex_space):
            for iy, ey in enumerate(ey_space):
                for it, eth in enumerate(eth_space):
                    for iv, v in enumerate(v_space):
                        for iw, w in enumerate(w_space):
                            tasks.append((t, ix, ex, iy, ey, it, eth, iv, v, iw, w, delta_t, sigma, ex_space, ey_space, eth_space, v_space, w_space))

    start_time = time.time()
    results = Parallel(n_jobs=-1)(delayed(compute_transition_matrix_for_state_control)(*task) for task in tqdm.tqdm(tasks))
    end_time = time.time()
    print(f"Time taken for parallel computation: {end_time - start_time:.2f} seconds")
    
    mean_next_state_total_time = 0
    find_neighbors_total_time = 0
    compute_prob_total_time = 0

    start_time = time.time()
    for res in results:
        t, ix, iy, it, iv, iw, local_transition_matrix = res
        transition_matrix[t, ix, iy, it, iv, iw, :, :] = local_transition_matrix

    for t in range(nt):
        for ix in range(len(ex_space)):
            for iy in range(len(ey_space)):
                for it in range(len(eth_space)):
                    for iv in range(len(v_space)):
                        for iw in range(len(w_space)):
                            total_prob = np.sum(transition_matrix[t, ix, iy, it, iv, iw, :, 3])
                            if total_prob > 0:
                                transition_matrix[t, ix, iy, it, iv, iw, :, 3] /= total_prob
    end_time = time.time()
    print(f"Time taken for normalization: {end_time - start_time:.2f} seconds")

# Example usage
sigma = np.array([0.04, 0.04, 0.004])
delta_t = 0.5
compute_transition_matrix(transition_matrix, ex_space, ey_space, eth_space, v_space, w_space, delta_t, sigma)
import pdb; pdb.set_trace()