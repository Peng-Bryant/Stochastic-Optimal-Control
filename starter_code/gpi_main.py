from time import time
import numpy as np
import utils
from GPI_final import GPI, GpiConfig
from discretization import create_state_space
import os
import argparse

def main():
    # Config GPI
    traj = utils.lissajous
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='GPI_pp4_1')
    parser.add_argument('--cp', type=str, default='10')
    exp = parser.parse_args().exp
    cp = parser.parse_args().cp
    Folder = f"./data/{exp}/"
    
    if not os.path.exists(Folder):
        os.makedirs(Folder)
    nt = 100
    ex_space, ey_space, eth_space, v_space, w_space = create_state_space()

    obstacles = np.array([[1.0, 2.0], [-2.0, -2.0]])
    Q = np.eye(2)  # State cost matrix
    q = 1.0  # Scalar cost for theta error
    R = np.eye(2)  # Control cost matrix
    gamma = 0.99  # Discount factor
    num_evals = 25  # Number of policy evaluations in each iteration
    collision_margin = 0.55  # Collision margin
    output_dir = f'./output/{exp}/'  # Output directory for saving results
    # output_dir = './output/'

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

    policy_folder = output_dir

    gpi = GPI(config, nt, batch_size=100)  # Adjust the batch size as needed
    gpi.load_policy(f"{policy_folder}policy_{cp}.npz")

    # Obstacles in the environment
    obstacles = np.array([[-2, -2, 0.5], [1, 2, 0.5]])
    # Params
    traj = utils.lissajous
    ref_traj = []
    error_trans = 0.0
    error_rot = 0.0
    car_states = []
    times = []
    # Start main loop
    main_loop = time()  # return time in sec
    # Initialize state
    cur_state = np.array([utils.x_init, utils.y_init, utils.theta_init])
    cur_iter = 0

    # Main loop
    while cur_iter * utils.time_step < utils.sim_time:
    # while cur_iter * utils.time_step < 20:

        t1 = time()
        # Get reference state
        cur_time = cur_iter * utils.time_step
        cur_ref = traj(cur_iter)
        # Save current state and reference state for visualization
        ref_traj.append(cur_ref)
        car_states.append(cur_state)

        ################################################################
        # Generate control input
        # TODO: Replace this simple controller with your own controller
        curr_time_step = cur_iter % utils.T
        error_state = cur_state - cur_ref
        control = gpi.policy_control(curr_time_step, error_state)
        print("[v,w]", control)
        ################################################################

        # Apply control input
        next_state = utils.car_next_state(utils.time_step, cur_state, control, noise=True)
        # Update current state
        cur_state = next_state
        # Loop time
        t2 = utils.time()
        print(cur_iter)
        print(t2 - t1)
        times.append(t2 - t1)
        cur_err = cur_state - cur_ref
        cur_err[2] = np.arctan2(np.sin(cur_err[2]), np.cos(cur_err[2]))
        error_trans = error_trans + np.linalg.norm(cur_err[:2])
        error_rot = error_rot + np.abs(cur_err[2])
        print(cur_err, error_trans, error_rot)
        print("======================")
        cur_iter = cur_iter + 1

    main_loop_time = time()
    print("\n\n")
    print("Total time: ", main_loop_time - main_loop)
    print("Average iteration time: ", np.array(times).mean() * 1000, "ms")
    print("Final error_trans: ", error_trans)
    print("Final error_rot: ", error_rot)

    # Visualization
    ref_traj = np.array(ref_traj)
    car_states = np.array(car_states)
    times = np.array(times)
    utils.visualize(car_states, ref_traj, obstacles, times, utils.time_step, save=True)

if __name__ == "__main__":
    main()

