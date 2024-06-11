import numpy as np

def adaptive_grid(error_threshold, fine_spacing, coarse_spacing):
    """
    Generate an adaptive grid based on the error state and error threshold.
    
    Args:
    - error_threshold (float): The threshold for switching between fine and coarse grids.
    - fine_spacing (float): The spacing for fine grid.
    - coarse_spacing (float): The spacing for coarse grid.
    
    Returns:
    - grid (np.ndarray): The combined adaptive grid.
    """
    fine_grid = np.arange(-error_threshold, error_threshold, fine_spacing)
    coarse_grid_low = np.arange(-3, -error_threshold, coarse_spacing)
    coarse_grid_high = np.arange(error_threshold, 3, coarse_spacing)
    grid = np.union1d(np.union1d(fine_grid, coarse_grid_low), coarse_grid_high)
    return grid

def adaptive_grid_xy():
    """
    Generate an adaptive grid based on the error state and error threshold.
    3 level grids for x and y
    Whole state space is [-3, 3]

    for [-0.25, 0.25] divided into 10 parts
    for [-0.5, 0.5] divided into 5 parts
    for [-3, 3] divided into 5 parts
    """
    # Fine grid for [-0.25, 0.25] divided into 10 parts
    fine_grid = np.linspace(-0.25, 0.25, 11)
    
    # Medium grid for [-0.5, 0.5] divided into 5 parts excluding the overlap with fine grid
    medium_grid = np.concatenate([
        np.linspace(-0.5, -0.25, 3),
        np.linspace(0.25, 0.5, 3)
    ])
    
    # Coarse grid for [-3, 3] divided into 5 parts excluding the overlap with medium and fine grid
    coarse_grid = np.concatenate([
        np.linspace(-3, -0.5, 6),
        np.linspace(0.5, 3, 6)
    ])
    
    # Combine all grids
    grid = np.sort(np.unique(np.concatenate([fine_grid, medium_grid, coarse_grid])))
    
    return grid

def adaptive_grid_theta():
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
        np.linspace(-np.pi/2, -np.pi/4, 3),
        np.linspace(np.pi/4, np.pi/2, 3)
    ])
    
    # Coarse grid for [-pi, pi] divided into 5 parts excluding the overlap with medium and fine grid
    coarse_grid = np.concatenate([
        np.linspace(-np.pi, -np.pi/2, 6),
        np.linspace(np.pi/2, np.pi, 6)
    ])
    
    # Combine all grids
    grid = np.sort(np.unique(np.concatenate([fine_grid, medium_grid, coarse_grid])))
    
    return grid


def adaptive_control_grid_w():
    """
    Generate an adaptive control grid based on the error state.
    3 level grids for theta
    Whole state space is [-1,1]

    for [-0.25,0.25] divided into 4 parts
    for [-0.6, 0.6] divided into 3 parts
    for [-1, 1] divided into 3 parts
    """
    # Fine grid for [-0.25, 0.25] divided into 4 parts
    fine_grid = np.linspace(-0.25, 0.25, 5)
    
    # Medium grid for [-0.6, 0.6] divided into 3 parts excluding the overlap with fine grid
    medium_grid = np.concatenate([
        np.linspace(-0.6, -0.25, 2),
        np.linspace(0.25, 0.6, 2)
    ])
    
    # Coarse grid for [-1, 1] divided into 3 parts excluding the overlap with medium and fine grid
    coarse_grid = np.concatenate([
        np.linspace(-1, -0.6, 2),
        np.linspace(0.6, 1, 2)
    ])
    
    # Combine all grids
    grid = np.sort(np.unique(np.concatenate([fine_grid, medium_grid, coarse_grid])))
    
    return grid

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
    fine_grid = np.linspace(0, 0.25, 5)
    
    # Medium grid for [0.25, 0.6] divided into 3 parts excluding the overlap with fine grid
    medium_grid = np.concatenate([
        np.linspace(0.25, 0.6, 3)
    ])
    
    # Coarse grid for [0.6, 1] divided into 3 parts excluding the overlap with medium and fine grid
    coarse_grid = np.concatenate([
        np.linspace(0.6, 1, 3)
    ])
    
    # Combine all grids
    grid = np.sort(np.unique(np.concatenate([fine_grid, medium_grid, coarse_grid])))
    
    return grid



def adaptive_control_grid(error_state, error_threshold, fine_spacing, coarse_spacing):
    """
    Generate an adaptive control grid based on the current error state.
    
    Args:
    - error_state (float): The current error state.
    - error_threshold (float): The threshold for switching between fine and coarse grids.
    - fine_spacing (float): The spacing for fine grid.
    - coarse_spacing (float): The spacing for coarse grid.
    
    Returns:
    - control_grid (np.ndarray): The adaptive control grid.
    """
    if abs(error_state) <= error_threshold:
        # Use fine control grid
        control_grid = np.arange(0, 1 + fine_spacing, fine_spacing)  # Example for v, adjust as needed
    else:
        # Use coarse control grid
        control_grid = np.arange(0, 1 + coarse_spacing, coarse_spacing)  # Example for v, adjust as needed
    return control_grid

def dynamic_R(self, error_state, obstacle_positions):
    """
    Adjust the R matrix based on the error state and proximity to obstacles.
    """
    base_R = np.diag([0.1, 0.1])  # Example base R matrix, adjust as needed
    increase_factor = 10  # Factor by which to increase R, adjust as needed
    min_distance_to_obstacle = np.min([np.linalg.norm(error_state[:2] - obs[:2]) for obs in obstacle_positions])
    
    if np.linalg.norm(error_state) <= 0.5 or min_distance_to_obstacle <= 0.5:
        return base_R * increase_factor
    else:
        return base_R
    


if __name__ == "__main__":
    # Example usage
    error_threshold = 0.5
    fine_spacing = 0.05
    coarse_spacing = 0.3

    ex_space = adaptive_grid(error_threshold, fine_spacing, coarse_spacing)  # 38
    ey_space = adaptive_grid(error_threshold, fine_spacing, coarse_spacing)    # 38
    eth_space = np.linspace(-np.pi, np.pi, 40)  # Angle grid can remain the same # 40
    import pdb; pdb.set_trace()
    # # Example usage
    # error_state = 0.3  # Example error state
    # error_threshold = 0.5
    # fine_spacing = 0.1
    # coarse_spacing = 0.5

    # v_space = adaptive_control_grid(error_state, error_threshold, fine_spacing, coarse_spacing)
    # w_space = np.linspace(-1, 1, int((2 / fine_spacing) + 1))  # Uniform grid for simplicity