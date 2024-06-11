import numpy as np

def adaptive_grid_xy():
    """
    Generate an adaptive grid based on the error state and error threshold.
    3 level grids for x and y
    Whole state space is [-2.5, 2.5]

    for [-0.25, 0.25] divided into 6 parts
    for [-0.5, 0.5] divided into 5 parts
    for [-1.5, 1.5] divided into 5 parts
    """
    # Fine grid for [-0.25, 0.25] divided into 6 parts
    fine_grid = np.linspace(-0.25, 0.25, 7)
    
    # Medium grid for [-0.5, 0.5] divided into 5 parts excluding the overlap with fine grid
    medium_grid = np.concatenate([
        np.linspace(-0.65, -0.25, 5),
        np.linspace(0.25, 0.65, 5)
    ])
    
    # Coarse grid for [-3, 3] divided into 5 parts excluding the overlap with medium and fine grid
    coarse_grid = np.concatenate([
        np.linspace(-1.5, -0.65, 6),
        np.linspace(0.65, 1.5, 6)
    ])
    
    # Combine all grids
    grid = np.sort(np.unique(np.concatenate([fine_grid, medium_grid, coarse_grid])))
    
    return grid

def adaptive_grid_theta_0():
    """
    Generate an adaptive grid based on the error state and error threshold.
    3 level grids for theta
    Whole state space is [-pi, pi]

    for [-pi/4, pi/4] divided into 10 parts
    for [-pi/2, pi/2] divided into 5 parts
    for [-pi, pi] divided into 5 parts
    """
    # Fine grid for [-pi/4, pi/4] divided into 10 parts
    fine_grid = np.linspace(-np.pi/4, np.pi/4, 11)
    
    # Medium grid for [-pi/2, pi/2] divided into 5 parts excluding the overlap with fine grid
    medium_grid = np.concatenate([
        np.linspace(-np.pi/2, -np.pi/4, 4),
        np.linspace(np.pi/4, np.pi/2, 4)
    ])
    
    # Coarse grid for [-pi, pi] divided into 5 parts excluding the overlap with medium and fine grid
    coarse_grid = np.concatenate([
        np.linspace(-np.pi, -np.pi/2, 7),
        np.linspace(np.pi/2, np.pi, 7)
    ])
    
    # Combine all grids
    grid = np.sort(np.unique(np.concatenate([fine_grid, medium_grid, coarse_grid])))
    
    return grid

def adaptive_grid_theta_1():
    """
    Generate an adaptive grid based on the error state and error threshold.
    3 level grids for theta
    Whole state space is [-pi, pi]

    for [-pi/4, pi/4] divided into 10 parts
    for [-pi/2, pi/2] divided into 5 parts
    for [-pi, pi] divided into 5 parts
    """
    # Fine grid for [-pi/4, pi/4] divided into 10 parts
    fine_grid = np.linspace(-np.pi/4, np.pi/4, 14)
    
    # Medium grid for [-pi/2, pi/2] divided into 5 parts excluding the overlap with fine grid
    medium_grid = np.concatenate([
        np.linspace(-np.pi/2, -np.pi/4, 5),
        np.linspace(np.pi/4, np.pi/2, 5)
    ])
    
    # Coarse grid for [-pi, pi] divided into 5 parts excluding the overlap with medium and fine grid
    coarse_grid = np.concatenate([
        np.linspace(-np.pi, -np.pi/2, 10),
        np.linspace(np.pi/2, np.pi, 10)
    ])
    
    # Combine all grids
    grid = np.sort(np.unique(np.concatenate([fine_grid, medium_grid, coarse_grid])))
    
    return grid

def adaptive_control_grid_w_0():
    """
    Generate an adaptive control grid based on the error state.
    3 level grids for theta
    Whole state space is [-1,1]

    for [-0.25,0.25] divided into 4 parts
    for [-0.6, 0.6] divided into 3 parts
    for [-1, 1] divided into 3 parts
    """
    # Fine grid for [-0.25, 0.25] divided into 4 parts
    fine_grid = np.linspace(-0.25, 0.25, 7)
    
    # Medium grid for [-0.6, 0.6] divided into 3 parts excluding the overlap with fine grid
    medium_grid = np.concatenate([
        np.linspace(-0.6, -0.25, 3),
        np.linspace(0.25, 0.6, 3)
    ])
    
    # Coarse grid for [-1, 1] divided into 3 parts excluding the overlap with medium and fine grid
    coarse_grid = np.concatenate([
        np.linspace(-1, -0.6, 3),
        np.linspace(0.6, 1, 3)
    ])
    
    # Combine all grids
    grid = np.sort(np.unique(np.concatenate([fine_grid, medium_grid, coarse_grid])))
    
    return grid

def adaptive_control_grid_w_1():
    return np.linspace(-1, 1, 10)

def adaptive_control_grid_v():
    """
    Generate an adaptive control grid based on the error state.
    3 level grids for theta
    Whole state space is [0,1]

    for [0,0.25] divided into 4 parts
    for [0.25, 0.6] divided into 3 parts
    for [0.6, 1] divided into 3 parts
    """
    # Fine grid for [0, 0.25] divided into 4 parts
    fine_grid = np.linspace(0, 0.3, 4)
    
    # Medium grid for [0.25, 0.6] divided into 3 parts excluding the overlap with fine grid
    medium_grid = np.concatenate([
        np.linspace(0.23, 0.6, 3)
    ])
    
    # Coarse grid for [0.6, 1] divided into 3 parts excluding the overlap with medium and fine grid
    coarse_grid = np.concatenate([
        np.linspace(0.6, 1, 3)
    ])
    
    # Combine all grids
    grid = np.sort(np.unique(np.concatenate([fine_grid, medium_grid, coarse_grid])))
    
    return grid

def adaptive_grid_theta_2():
    """the finest-level resolution is the evenly distance
      from −π/4 to π/4 divided into 5 parts; the middle-level 
      is descretized to ±π/2; and the coarsest-level, ±π"""
    # Fine grid for [-pi/4, pi/4] divided into 5 parts
    fine_grid = np.linspace(-np.pi/4, np.pi/4, 6)
    
    # Medium grid for [-pi/2, pi/2] divided into 5 parts excluding the overlap with fine grid
    medium_grid = np.concatenate([
        np.linspace(-np.pi/2, -np.pi/4, 3),
        np.linspace(np.pi/4, np.pi/2, 3)
    ])
    
    # Coarse grid for [-pi, pi] divided into 5 parts excluding the overlap with medium and fine grid
    coarse_grid = np.concatenate([
        np.linspace(-np.pi, -np.pi/2, 4),
        np.linspace(np.pi/2, np.pi, 4)
    ])
    
    # Combine all grids
    grid = np.sort(np.unique(np.concatenate([fine_grid, medium_grid, coarse_grid])))
    
    return grid
    
def create_state_space_():
    """
    Create the state space for the system.
    """
    # Define the state space_0
    ex_space = adaptive_grid_xy()
    ey_space = adaptive_grid_xy()
    eth_space = adaptive_grid_theta_0()
    v_space = adaptive_control_grid_v()
    w_space = adaptive_control_grid_w_0()
 
    print("State space created! Shape of state space is: ", ex_space.shape, ey_space.shape, eth_space.shape, v_space.shape, w_space.shape)
    return ex_space, ey_space, eth_space, v_space, w_space


def create_state_space_():
    """
    Create the state space 1 for the system.
    """
    # Define the state space_1
    ex_space = adaptive_grid_xy()
    ey_space = adaptive_grid_xy()
    eth_space = adaptive_grid_theta_1()
    v_space = adaptive_control_grid_v()
    w_space = adaptive_control_grid_w_1()
 
    print("State space 1 created! Shape of state space is: ", ex_space.shape, ey_space.shape, eth_space.shape, v_space.shape, w_space.shape)
    return ex_space, ey_space, eth_space, v_space, w_space


def create_state_space():
    """
    Create the state space 2 for the system.
    """
    ex_space = adaptive_grid_xy()
    ey_space = adaptive_grid_xy()
    eth_space = adaptive_grid_theta_2()
    v_space =  np.linspace(0, 1, 11)
    w_space = np.linspace(-1, 1, 11)

    print("State space 2 created! Shape of state space is: ", ex_space.shape, ey_space.shape, eth_space.shape, v_space.shape, w_space.shape)
    return ex_space, ey_space, eth_space, v_space, w_space




if __name__ == "__main__":
    create_state_space()
    pass